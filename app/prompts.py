import re

from app.models import AnalysisRequest, AnalysisResponse, build_sourced_response_type

# Separate agent's identity from the specific task
SYSTEM_PROMPT = (
    "You are a competitive intelligence analyst. Given a competitor's name "
    "(and optionally its official website and a focus area), produce a "
    "structured analysis. Be concrete; prefer stating information is "
    "unavailable over inventing figures. Before including a claim, confirm "
    "it's supported by the research findings provided. If you can't find "
    "support for a specific detail, omit it or state it's unavailable "
    "rather than approximating."
)


def build_user_message(request: AnalysisRequest) -> str:
    lines = [f"Analyze the competitor: {request.competitor_name}"]
    # url/focus are optional on the request, so only include what's present.
    if request.competitor_url:
        lines.append(f"Official website (for disambiguation): {request.competitor_url}")
    if request.focus:
        lines.append(f"Focus specifically on: {request.focus}")
    return "\n".join(lines)


# Separate from SYSTEM_PROMPT: this drives research turns (tool use, no
# output schema); SYSTEM_PROMPT still drives only the final synthesis call.
RESEARCH_SYSTEM_PROMPT = (
    "You are a competitive intelligence analyst gathering information about "
    "a competitor using web search. For questions where current information "
    "would change the answer, search before answering rather than answering "
    "from memory. Search for products, pricing signals, regional presence, "
    "and recent news; stop searching once you have enough to cover those "
    "areas."
)


# Matches the title line dispatch() writes per result: "TITLE (URL)". Anchored
# to the end of the line, not searched anywhere in the text, so a URL merely
# quoted inside a page's own content can't be mistaken for a searched source.
_URL_IN_TITLE_LINE = re.compile(r"\((https?://[^\s)]+)\)\s*$")


def extract_source_urls(findings: list[tuple[str, str]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for _, result in findings:
        for block in result.split("\n\n"):
            first_line = block.split("\n", 1)[0]
            match = _URL_IN_TITLE_LINE.search(first_line)
            if match and match.group(1) not in seen:
                seen.add(match.group(1))
                urls.append(match.group(1))
    return urls


def build_synthesis_message(
    user_message: str, findings: list[tuple[str, str]]
) -> tuple[str, type[AnalysisResponse]]:
    if not findings:
        return user_message + "\n\n(No search results were gathered.)", AnalysisResponse

    urls = extract_source_urls(findings)
    sections = [f"Search: {query}\nFindings: {result}" for query, result in findings]
    sources = "\n".join(f"- {url}" for url in urls)
    message = (
        user_message
        + "\n\nResearch findings:\n\n"
        + "\n\n".join(sections)
        + "\n\nAvailable sources (cite only these, exactly as written):\n"
        + sources
    )
    # response_type is dynamically narrowed to this run's real URLs — the
    # schema itself won't accept a source_url that wasn't actually found.
    response_type = build_sourced_response_type(urls) if urls else AnalysisResponse
    return message, response_type
