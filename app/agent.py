from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock
from langfuse import get_client

from app.diagnostics import ResponseWarning, SearchTally, make_warning, partial_search_failure
from app.llm import LLMClient
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
    user_message = build_user_message(request)
    messages: list[MessageParam] = [{"role": "user", "content": user_message}]
    tally = SearchTally()

    # Scoped to just the research phase, so the final synthesis generation
    # nests as a sibling of this span under `analyze`, not inside it —
    # they're two distinct jobs (research vs. write-the-report).
    with get_client().start_as_current_observation(name="agent_loop", as_type="span") as loop_span:
        stop_reason: str | None = None
        for _ in range(MAX_ITERATIONS):
            response = llm_client.complete_with_tools(RESEARCH_SYSTEM_PROMPT, messages, tools=[SEARCH_TOOL])
            messages.append({"role": "assistant", "content": response.content})
            stop_reason = response.stop_reason
            if stop_reason != "tool_use":
                break
            tool_results = [dispatch(block, tally) for block in response.content if isinstance(block, ToolUseBlock)]
            messages.append({"role": "user", "content": tool_results})
        # cap_hit is derivable from stop_reason, but spelling it out saves a
        # reader of the trace from having to remember what "tool_use" means here.
        loop_span.update(metadata={"stop_reason": stop_reason, "cap_hit": stop_reason == "tool_use"})

    warnings: list[ResponseWarning] = []
    if stop_reason == "tool_use":
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
