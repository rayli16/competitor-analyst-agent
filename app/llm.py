import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from anthropic import (
    Anthropic,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
)
from anthropic.types import Message, MessageParam, ToolParam
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel

from app.diagnostics import FallbackTally

# Reads .env into the process environment; safe to call even if .env is missing.
load_dotenv()

logger = logging.getLogger(__name__)


def get_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fail fast here with a clear message, instead of a confusing
        # AuthenticationError later inside an API call.
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return api_key


T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


@dataclass(frozen=True)
class ModelEndpoint:
    client: Anthropic
    model: str
    # Overrides any effort the caller asks for, on every call to this endpoint.
    effort: str | None = None

    def output_config(self, requested: dict | None = None) -> dict | None:
        if self.effort is None:
            return requested
        return {**(requested or {}), "effort": self.effort}


# Kimi through Moonshot's Anthropic-compatible endpoint, so the same SDK calls
# (tool use, messages.parse) work unchanged. A different provider on purpose:
# a fallback to another Claude model would go down with Anthropic in an outage.
MOONSHOT_BASE_URL = "https://api.moonshot.ai/anthropic"
# Measured on one real report (9 searches of findings): kimi-k3.1-pro at its
# default effort (max) didn't finish in 6 min; at low effort it took 80s, and
# the highspeed variant at low effort 34s, with reports comparable to Claude's.
MOONSHOT_MODEL = "kimi-k3.1-pro-highspeed"
# Kimi accepts low/high/max (no "medium"), and the Anthropic-style
# output_config.effort is the form its Anthropic endpoint honours —
# reasoning_effort from its OpenAI-style docs had no effect there.
MOONSHOT_EFFORT = "low"


def get_fallback_endpoint() -> ModelEndpoint | None:
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        return None  # optional: without it, a primary failure fails the request as before
    # Keys from the China platform (platform.moonshot.cn) only work on
    # https://api.moonshot.cn/anthropic; set MOONSHOT_BASE_URL for those.
    base_url = os.getenv("MOONSHOT_BASE_URL") or MOONSHOT_BASE_URL
    # The SDK default (10 min per attempt, 2 retries) let one stalled report
    # call hold a request for up to 30 min. A fallback that slow is worse
    # than failing, so give up after 3 min and one retry.
    client = Anthropic(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=1)
    return ModelEndpoint(client, os.getenv("MOONSHOT_MODEL") or MOONSHOT_MODEL, effort=MOONSHOT_EFFORT)


# Failures that another provider could serve: an outage, overload, rate
# limit, network error, retired model, or a bad/blocked key. Not 400/422 —
# a malformed request is our bug and would fail on the fallback too.
FALLBACK_ERRORS = (
    APIConnectionError,  # includes APITimeoutError
    InternalServerError,
    OverloadedError,
    RateLimitError,
    NotFoundError,
    AuthenticationError,
    PermissionDeniedError,
)

_fallbacks: ContextVar[FallbackTally | None] = ContextVar("llm_fallbacks", default=None)


@contextmanager
def track_fallbacks() -> Iterator[FallbackTally]:
    """Scopes fallback state to one request. Worker threads started through
    map_in_context see the same tally, since the context is copied on the
    calling thread."""
    tally = FallbackTally()
    token = _fallbacks.set(tally)
    try:
        yield tally
    finally:
        _fallbacks.reset(token)


class LLMClient:
    def __init__(
        self,
        primary: ModelEndpoint | None = None,
        fallback: ModelEndpoint | None = None,
        max_tokens: int = 12000,
    ) -> None:
        # The SDK already retries connection errors, 408/409/429 and 5xx with
        # backoff; set explicitly so the fallback below clearly runs only
        # after those retries are spent.
        self._primary = primary or ModelEndpoint(Anthropic(api_key=get_api_key(), max_retries=2), "claude-sonnet-5")
        self._fallback = fallback
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._primary.model

    def _call(self, send: Callable[[ModelEndpoint], R]) -> tuple[R, str]:
        tally = _fallbacks.get()
        if self._fallback is None or tally is None or not tally.switched:
            try:
                return send(self._primary), self._primary.model
            except FALLBACK_ERRORS as e:
                if self._fallback is None:
                    raise
                logger.warning("Primary model failed (%s); falling back to %s", type(e).__name__, self._fallback.model)
                if tally is not None:
                    tally.record(e, self._fallback.model)
        # Once switched, stay switched for the rest of the request: during an
        # outage each primary attempt costs its retries' worth of waiting.
        return send(self._fallback), self._fallback.model

    @staticmethod
    def _config(endpoint: ModelEndpoint, requested: dict | None = None) -> dict:
        config = endpoint.output_config(requested)
        return {"output_config": config} if config is not None else {}

    def complete(self, system_prompt: str, user_message: str) -> str:
        response, _ = self._call(
            lambda ep: ep.client.messages.create(
                model=ep.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                **self._config(ep),
            )
        )
        # content is a list of blocks (text, tool-use, ...); this assumes a
        # single text block, which holds for a plain text completion.
        return response.content[0].text

    def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_format: type[T],
        output_config: dict | None = None,
    ) -> T:
        # get_client() returns a cached singleton, cheap to call per-request.
        with get_client().start_as_current_observation(
            name="complete_structured",
            as_type="generation",
            model=self.model,
            input=user_message,
        ) as gen:
            # output_format's generated schema gets merged into output_config
            # by the SDK, not overwritten, so passing an explicit effort here
            # doesn't clobber the schema constraint.
            response, model = self._call(
                lambda ep: ep.client.messages.parse(
                    model=ep.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    output_format=output_format,
                    **self._config(ep, output_config),
                )
            )
            parsed = response.parsed_output
            if parsed is None:
                # e.g. stop_reason == "refusal" — no text block to parse.
                # Raised inside the `with` block so OTel marks this
                # generation as errored automatically; no manual status code.
                raise RuntimeError(f"Model produced no structured output (stop_reason={response.stop_reason!r}).")
            gen.update(
                model=model,
                output=parsed.model_dump(),
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            return parsed

    def complete_with_tools(self, system_prompt: str, messages: list[MessageParam], tools: list[ToolParam]) -> Message:
        # No explicit `input=` here — messages grows every loop iteration,
        # and re-embedding the whole history on each turn's trace would be
        # redundant with what each prior turn's own span already shows.
        with get_client().start_as_current_observation(
            name="complete_with_tools",
            as_type="generation",
            model=self.model,
        ) as gen:
            # A mid-loop switch hands the fallback Claude's earlier turns,
            # thinking blocks included; Kimi accepts them (checked by hand).
            response, model = self._call(
                lambda ep: ep.client.messages.create(
                    model=ep.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    **self._config(ep),
                )
            )
            gen.update(
                model=model,
                output=[block.model_dump() for block in response.content],
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            return response
