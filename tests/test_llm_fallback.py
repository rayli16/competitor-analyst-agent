"""Cross-provider fallback: when the primary model fails in a way another
provider could serve, the fallback takes over for the rest of the request and
the response says so. Model clients are faked with injected faults, so these
run offline at no cost."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)

from app import agent
from app.llm import LLMClient, ModelEndpoint, track_fallbacks
from app.models import AnalysisRequest
from tests.test_warnings import FakeAgentLLM, FakeSearch, codes

REQUEST = httpx.Request("POST", "https://api.example/v1/messages")


def status_error(cls, code: int):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


class FlakyMessages:
    """messages.create / messages.parse that raise the queued faults first,
    then answer from `reply`."""

    def __init__(self, faults, reply):
        self.faults, self.reply, self.calls = list(faults), reply, 0

    def _next(self, **kwargs):
        self.calls += 1
        if self.faults:
            raise self.faults.pop(0)
        return self.reply(**kwargs)

    create = _next
    parse = _next


def endpoint(model: str, faults=(), reply=None) -> ModelEndpoint:
    reply = reply or (lambda **kw: SimpleNamespace(model=kw["model"]))
    return ModelEndpoint(SimpleNamespace(messages=FlakyMessages(faults, reply)), model)


def calls(ep: ModelEndpoint) -> int:
    return ep.client.messages.calls


def test_primary_success_never_touches_fallback():
    primary, fallback = endpoint("claude"), endpoint("kimi")
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks() as tally:
        _, model = llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    assert model == "claude"
    assert calls(fallback) == 0 and not tally.switched


@pytest.mark.parametrize(
    "fault",
    [
        status_error(InternalServerError, 500),
        status_error(OverloadedError, 529),
        status_error(RateLimitError, 429),
        APIConnectionError(request=REQUEST),
    ],
)
def test_serveable_failures_fall_back(fault):
    primary, fallback = endpoint("claude", [fault]), endpoint("kimi")
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks() as tally:
        _, model = llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    assert model == "kimi"
    assert tally.errors == {type(fault).__name__: 1} and tally.model == "kimi"


def test_after_a_switch_the_rest_of_the_request_skips_the_primary():
    # Waiting out the primary's retries on every call during an outage would
    # multiply the delay; one failure is enough to switch.
    primary, fallback = endpoint("claude", [status_error(OverloadedError, 529)]), endpoint("kimi")
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks():
        for _ in range(3):
            llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    assert (calls(primary), calls(fallback)) == (1, 3)


def test_a_new_request_tries_the_primary_again():
    primary, fallback = endpoint("claude", [status_error(OverloadedError, 529)]), endpoint("kimi")
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks():
        llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    with track_fallbacks() as second:
        _, model = llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    assert model == "claude" and not second.switched


def test_fallback_effort_overrides_the_callers_and_leaves_claude_alone():
    # Kimi at its default (max) effort couldn't write a report in 6 min, and
    # it has no "medium" level, so its endpoint effort replaces the caller's.
    sent = []

    class Out(BaseModel):
        pass

    def reply(**kwargs):
        sent.append((kwargs["model"], kwargs.get("output_config")))
        return SimpleNamespace(parsed_output=Out(), stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    primary = endpoint("claude", reply=reply)
    fallback = ModelEndpoint(endpoint("kimi", reply=reply).client, "kimi", effort="low")
    llm = LLMClient(primary=primary, fallback=fallback)
    llm.complete_structured("s", "m", Out, output_config={"effort": "medium"})
    primary.client.messages.faults = [status_error(OverloadedError, 529)]
    with track_fallbacks():
        llm.complete_structured("s", "m", Out, output_config={"effort": "medium"})
    assert sent == [("claude", {"effort": "medium"}), ("kimi", {"effort": "low"})]


def test_bad_request_is_our_bug_and_does_not_fall_back():
    primary, fallback = endpoint("claude", [status_error(BadRequestError, 400)]), endpoint("kimi")
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks(), pytest.raises(BadRequestError):
        llm._call(lambda ep: ep.client.messages.create(model=ep.model))
    assert calls(fallback) == 0


def test_without_a_fallback_the_original_error_surfaces():
    llm = LLMClient(primary=endpoint("claude", [status_error(OverloadedError, 529)]))
    with track_fallbacks(), pytest.raises(OverloadedError):
        llm._call(lambda ep: ep.client.messages.create(model=ep.model))


def test_fallback_failing_too_raises_its_error():
    primary = endpoint("claude", [status_error(OverloadedError, 529)])
    fallback = endpoint("kimi", [status_error(InternalServerError, 500)])
    llm = LLMClient(primary=primary, fallback=fallback)
    with track_fallbacks(), pytest.raises(InternalServerError):
        llm._call(lambda ep: ep.client.messages.create(model=ep.model))


# ---- what the client sees when no model could answer ----


@pytest.mark.parametrize(
    ("fault", "status"),
    [
        (APITimeoutError(request=REQUEST), 504),
        (status_error(OverloadedError, 529), 503),
        (APIConnectionError(request=REQUEST), 503),
        (status_error(BadRequestError, 400), 500),
    ],
)
def test_model_failures_map_to_clear_http_errors(monkeypatch, fault, status):
    from fastapi.testclient import TestClient

    from app import main

    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(main, "run_agentic_analysis", fail)
    response = TestClient(main.app, raise_server_exceptions=False).post("/analyze", json={"competitor_name": "Acme"})
    assert response.status_code == status
    assert "model service" in response.json()["detail"]


# ---- end to end through /analyze ----


def test_analysis_reports_model_fallback(monkeypatch):
    monkeypatch.setattr(agent, "tavily_client", FakeSearch())
    script = FakeAgentLLM(stop_after=1)  # one search turn, then finish_research

    def tool_turn(**kwargs):
        response = script.complete_with_tools(kwargs["system"], kwargs["messages"], kwargs["tools"])
        return SimpleNamespace(content=response.content, usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    def report(**kwargs):
        parsed = script.complete_structured(kwargs["system"], kwargs["messages"][0]["content"], kwargs["output_format"])
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    primary = endpoint("claude", [status_error(OverloadedError, 529)] * 10)
    fallback = ModelEndpoint(SimpleNamespace(messages=SimpleNamespace(create=tool_turn, parse=report)), "kimi")
    result = agent.run_agentic_analysis(LLMClient(primary=primary, fallback=fallback), AnalysisRequest(competitor_name="Acme"))

    assert script.synthesized and result.products
    assert codes(result.warnings) == ["model_fallback"]
    w = result.warnings[0]
    assert w.severity == "info"
    assert w.details.model == "kimi" and w.details.error_types == {"OverloadedError": 1}
    assert calls(primary) == 1  # only the first turn tried Claude


def test_clean_analysis_has_no_fallback_warning(monkeypatch):
    # FakeAgentLLM bypasses LLMClient entirely, so the tally stays empty.
    monkeypatch.setattr(agent, "tavily_client", FakeSearch())
    result = agent.run_agentic_analysis(FakeAgentLLM(stop_after=1), AnalysisRequest(competitor_name="Acme"))
    assert "model_fallback" not in codes(result.warnings)
