import os
from typing import TypeVar

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel

# Reads .env into the process environment; safe to call even if .env is missing.
load_dotenv()


def get_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fail fast here with a clear message, instead of a confusing
        # AuthenticationError later inside an API call.
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return api_key


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        max_tokens: int = 12000,
    ) -> None:
        # api_key override lets callers/tests inject a key without touching .env
        self._client = Anthropic(api_key=api_key or get_api_key())
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
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
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                output_format=output_format,
                **({"output_config": output_config} if output_config is not None else {}),
            )
            parsed = response.parsed_output
            if parsed is None:
                # e.g. stop_reason == "refusal" — no text block to parse.
                # Raised inside the `with` block so OTel marks this
                # generation as errored automatically; no manual status code.
                raise RuntimeError(f"Model produced no structured output (stop_reason={response.stop_reason!r}).")
            gen.update(
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
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            gen.update(
                output=[block.model_dump() for block in response.content],
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            return response
