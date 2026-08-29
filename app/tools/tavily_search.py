import os

from anthropic.types import ToolParam
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


def get_tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # Unlike our own fail-fast, TavilyClient(api_key=None) would silently
        # fall back to a rate-limited "keyless" mode instead of raising.
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
    return api_key


class TavilySearchClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = TavilyClient(api_key=api_key or get_tavily_api_key())

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        response = self._client.search(query, search_depth="advanced", max_results=max_results)
        # Trim to what the agent actually needs; drop score/raw_content/id.
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response["results"]
        ]


class SearchInput(BaseModel):
    # dispatch() validates each tool call's raw input through this model
    # before calling Tavily — one schema, used for both steering and validation.
    query: str = Field(description="The search query to run.")


SEARCH_TOOL: ToolParam = {
    "name": "search",
    "description": (
        "Search the web for current, real information about a competitor — "
        "products, pricing, market position, recent news. Use this before "
        "writing the analysis rather than relying on prior knowledge, since "
        "competitor information changes and prior knowledge may be outdated "
        "or incomplete."
    ),
    "input_schema": SearchInput.model_json_schema(),
}
