from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock
from langfuse import get_client

from app.diagnostics import ResponseWarning, SearchTally, make_warning, model_fallback, partial_search_failure
from app.discovery import map_in_context
from app.llm import LLMClient, track_fallbacks
from app.models import AnalysisReport, AnalysisRequest
from app.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_synthesis_message,
    build_user_message,
    extract_source_urls,
)
from app.tools.tavily_search import SEARCH_TOOL, SearchInput, TavilySearchClient

MAX_ITERATIONS = 5  # safety cap, not tuned — revisit once Langfuse shows real iteration counts

# Bounds the findings synthesis has to read. Claude issues 1-3 searches per
# turn; Kimi issued 4-8, and a run reached 38 searches / 210k chars of
# findings — Kimi then ran 400s and hit max_tokens mid-JSON writing the report.
MAX_SEARCHES_PER_TURN = 3

# How the model ends research. Without it, the last turn ends on plain text:
# Langfuse showed that turn writing a full 2,000-2,700-token analysis every
# time — ~25s per run — which synthesis then discards and rewrites as the
# structured report. A tool call ends the loop in a few tokens instead.
FINISH_TOOL: ToolParam = {
    "name": "finish_research",
    "description": (
        "Call this once your searches cover the competitor's products, pricing signals, "
        "regional presence, and recent news. It ends research. The report is written "
        "separately from your search results, so don't write any analysis or summary."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

tavily_client = TavilySearchClient()


def dispatch(block: ToolUseBlock, tally: SearchTally) -> ToolResultBlockParam:
    with get_client().start_as_current_observation(
        name=block.name,
        as_type="tool",
        input=block.input,
    ) as tool_span:
        try:
            search_input = SearchInput.model_validate(block.input)
            results = tavily_client.search(search_input.query)
            content = "\n\n".join(f"{r['title']} ({r['url']})\n{r['content']}" for r in results)
        except Exception as e:
            # Malformed input and Tavily-side failures both become a normal
            # tool_result the model can see and react to, not a crash. This
            # is the one place in the codebase that catches broadly on
            # purpose: whatever went wrong, the recovery is identical
            # (is_error=True). The type name is kept only for the warning.
            tool_span.update(output=str(e), level="ERROR", status_message=str(e))
            tally.record(e)
            return {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}
        tally.record()
        tool_span.update(output=content)
        return {"type": "tool_result", "tool_use_id": block.id, "content": content}


def extract_findings(messages: list[MessageParam]) -> list[tuple[str, str]]:
    # Maps each tool_use_id to its query, so results pair correctly even if
    # multiple tools ran in one turn — list position isn't reliable.
    queries: dict[str, str] = {}
    findings: list[tuple[str, str]] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue  # the initial user turn is a plain string, not blocks
        for block in content:
            if isinstance(block, ToolUseBlock):
                queries[block.id] = block.input.get("query", "")
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                findings.append((queries.get(block["tool_use_id"], ""), block["content"]))
    return findings


def run_agentic_analysis(llm_client: LLMClient, request: AnalysisRequest) -> AnalysisReport:
    with track_fallbacks() as fallbacks:
        report = _run_agentic_analysis(llm_client, request)
    if fallbacks.switched:
        report.warnings.append(model_fallback(fallbacks))
    return report


def _run_agentic_analysis(llm_client: LLMClient, request: AnalysisRequest) -> AnalysisReport:
    user_message = build_user_message(request)
    messages: list[MessageParam] = [{"role": "user", "content": user_message}]
    tally = SearchTally()

    # Scoped to just the research phase, so the final synthesis generation
    # nests as a sibling of this span under `analyze`, not inside it —
    # they're two distinct jobs (research vs. write-the-report).
    with get_client().start_as_current_observation(name="agent_loop", as_type="span") as loop_span:
        finished_by: str | None = None
        for _ in range(MAX_ITERATIONS):
            response = llm_client.complete_with_tools(
                RESEARCH_SYSTEM_PROMPT, messages, tools=[SEARCH_TOOL, FINISH_TOOL]
            )
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [block for block in response.content if isinstance(block, ToolUseBlock)]
            if not tool_uses:
                # Stopped without calling finish_research. Still a finish,
                # just the expensive kind: the turn is plain-text output.
                finished_by = "end_turn"
                break
            searches = [block for block in tool_uses if block.name != FINISH_TOOL["name"]]
            if searches:
                # Searches issued alongside finish_research still run — their
                # results feed the report. No further model turn follows a
                # finish, so finish_research itself never needs a tool_result.
                run, skipped = searches[:MAX_SEARCHES_PER_TURN], searches[MAX_SEARCHES_PER_TURN:]
                results = map_in_context(lambda block: dispatch(block, tally), run, max_workers=len(run))
                # Every tool_use still needs a tool_result, so over-limit
                # calls get an error the model can see and re-issue later.
                results += [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Not run: at most {MAX_SEARCHES_PER_TURN} searches per turn.",
                        "is_error": True,
                    }
                    for block in skipped
                ]
                messages.append({"role": "user", "content": results})
            if len(searches) < len(tool_uses):
                finished_by = "finish_tool"
                break
        loop_span.update(metadata={"finished_by": finished_by, "cap_hit": finished_by is None})

    warnings: list[ResponseWarning] = []
    if finished_by is None:
        warnings.append(
            make_warning(
                "iteration_cap_hit",
                f"Research stopped at the {MAX_ITERATIONS}-iteration cap while the model still wanted to "
                "search, so the report may be missing areas it hadn't covered yet.",
            )
        )
    if tally.failed:
        warnings.append(
            partial_search_failure("research", tally, "The report only covers what the successful searches found.")
        )

    findings = extract_findings(messages)
    if not extract_source_urls(findings):
        # No sources means nothing to cite, so no synthesis call at all —
        # cheaper, and the alternative (an unconstrained schema) would let
        # the model write sourced-looking claims from memory.
        warnings.append(
            make_warning(
                "no_sources",
                "No search returned any sources, so no report was generated. "
                "The empty sections mean missing data, not missing products or markets.",
                **tally.as_details("research"),
            )
        )
        return AnalysisReport(
            competitor_name=request.competitor_name,
            summary="No report: research gathered no sources. See warnings.",
            products=[],
            pricing_signals=[],
            regional_presence=[],
            recommended_actions=[],
            warnings=warnings,
        )

    # response_type is narrowed to this run's real discovered URLs — see
    # build_synthesis_message.
    synthesis_message, response_type = build_synthesis_message(user_message, findings)
    report = llm_client.complete_structured(SYSTEM_PROMPT, synthesis_message, response_type)
    return AnalysisReport(**report.model_dump(), warnings=warnings)
