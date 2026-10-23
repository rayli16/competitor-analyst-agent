import logging
import os
import threading
from collections.abc import Callable

from anthropic.types import ToolParam
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tavily.errors import ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError

load_dotenv()

logger = logging.getLogger(__name__)

# Failures another key can actually fix: plan/quota exhausted (403/432/433),
# rate limited (429), bad key (401). Anything else — timeout, bad request —
# would fail identically on every key, so it's raised without trying them.
KEY_SWITCHING_ERRORS = (ForbiddenError, UsageLimitExceededError, InvalidAPIKeyError)


def get_tavily_api_keys() -> list[str]:
    # TAVILY_API_KEYS is a comma-separated, ordered list: primary first,
    # fallbacks after. Plain TAVILY_API_KEY still works as a single key.
    raw = os.getenv("TAVILY_API_KEYS") or os.getenv("TAVILY_API_KEY") or ""
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        # Unlike our own fail-fast, TavilyClient(api_key=None) would silently
        # fall back to a rate-limited "keyless" mode instead of raising.
        raise RuntimeError("No Tavily key set. Add TAVILY_API_KEYS (or TAVILY_API_KEY) to your .env file.")
    return keys


class TavilySearchClient:
    def __init__(
        self,
        api_keys: list[str] | None = None,
        client_factory: Callable[..., TavilyClient] = TavilyClient,
    ) -> None:
        # client_factory lets tests inject fakes without real keys or network.
        self._clients = [client_factory(api_key=key) for key in (api_keys or get_tavily_api_keys())]
        # Index of the key that last worked. Shared by every concurrent search
        # in a discovery run, so once one thread finds key 1 exhausted, the
        # rest start from key 2 instead of each hitting the dead key first.
        self._active = 0
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        with self._lock:
            start = self._active
        # Starts from the last working key and wraps around, so a key that was
        # only rate limited (or refilled at the monthly reset) gets used again
        # once the later ones fail, without restarting the process.
        order = [(start + offset) % len(self._clients) for offset in range(len(self._clients))]
        last_error: Exception | None = None
        for index in order:
            try:
                response = self._clients[index].search(query, search_depth="advanced", max_results=max_results)
            except KEY_SWITCHING_ERRORS as e:
                # Key position only, never the key itself.
                logger.warning("Tavily key #%d of %d rejected (%s)", index + 1, len(self._clients), type(e).__name__)
                last_error = e
                continue
            if index != start:
                with self._lock:
                    self._active = index
                logger.warning("Tavily switched to key #%d of %d", index + 1, len(self._clients))
            # Trim to what the agent actually needs; drop score/raw_content/id.
            return [
                {"title": r["title"], "url": r["url"], "content": r["content"]}
                for r in response["results"]
            ]
        # Every key failed the same recoverable way — surface the real error
        # type so callers' existing handling (is_error / skip) still applies.
        raise last_error


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
