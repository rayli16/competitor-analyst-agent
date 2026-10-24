"""Partial-success reporting: every way a run can come back incomplete must
show up in `warnings` instead of passing as a complete result. Searches and
LLM calls are faked, so these run offline at no cost."""

import hashlib
from types import SimpleNamespace
from typing import get_args, get_origin

import pytest
from anthropic.types import ToolUseBlock
from tavily.errors import ForbiddenError

from app import agent, discovery
from app.diagnostics import SEVERITY, SearchTally, WarningCode
from app.discovery_models import (
    CandidateScore,
    CandidateVerdict,
    Company,
    DiscoveryRequest,
    DedupResult,
    ExtractionResult,
    ScoringResult,
)
from app.models import AnalysisRequest, AnalysisResponse

REQUEST = DiscoveryRequest(our_products="optometry equipment")


def codes(warnings) -> list[str]:
    return [w.code for w in warnings]


class FakeSearch:
    """Tavily stand-in. fail(query) decides per query whether to raise."""

    def __init__(self, fail=lambda query: False) -> None:
        self.fail = fail

    def search(self, query, max_results=5):
        if self.fail(query):
            raise ForbiddenError("plan usage limit")
        # One distinct site per query, derived from the query itself so it's
        # deterministic (hash() is salted per process). Full digest, not a
        # prefix: several real queries share their first 20 bytes.
        slug = hashlib.sha256(query.encode()).hexdigest()[:16]
        return [{"title": f"T{slug}", "url": f"https://s{slug}.example/page", "content": "c"}]


class FakeLLM:
    """Answers each discovery stage by output type. Every search result becomes
    one manufacturer verdict and one company; scoring gives every company the
    same score, except URLs listed in skip_scoring are left out."""

    def __init__(self, skip_scoring: set[str] = frozenset()) -> None:
        self.skip_scoring = skip_scoring

    def complete_structured(self, system_prompt, message, output_format, output_config=None):
        if issubclass(output_format, ExtractionResult):
            urls = _allowed_urls(output_format, "verdicts", "url")
            return ExtractionResult(
                verdicts=[
                    CandidateVerdict(company_name=u, url=u, is_own_page=True, entity_type="manufacturer", evidence="e")
                    for u in urls
                ]
            )
        if issubclass(output_format, DedupResult):
            urls = _allowed_urls(output_format, "companies", "urls")
            return DedupResult(
                companies=[
                    Company(canonical_name=f"Co {u}", urls=[u], entity_type="manufacturer", merge_reasoning="m")
                    for u in urls
                ]
            )
        if issubclass(output_format, ScoringResult):
            urls = _allowed_urls(output_format, "scores", "company_url")
            return ScoringResult(
                scores=[
                    CandidateScore(
                        company_url=u,
                        reasoning="r",
                        product_overlap=8,
                        export_orientation=8,
                        info_availability=8,
                        llm_recommendation="deep_dive",
                    )
                    for u in urls
                    if u not in self.skip_scoring
                ]
            )
        raise AssertionError(f"unexpected output_format {output_format}")


def _allowed_urls(output_format, list_field: str, url_field: str) -> list[str]:
    # The URLs a call may cite are the Literal values on its dynamic item
    # type — the same constraint the real model gets, read off the type.
    (item_type,) = get_args(output_format.model_fields[list_field].annotation)
    url_annotation = item_type.model_fields[url_field].annotation
    if get_origin(url_annotation) is list:  # Company.urls is list[Literal[...]]
        (url_annotation,) = get_args(url_annotation)
    return list(get_args(url_annotation))


@pytest.fixture
def search(monkeypatch):
    def install(fake: FakeSearch) -> None:
        monkeypatch.setattr(discovery, "tavily_client", fake)
        monkeypatch.setattr(agent, "tavily_client", fake)

    return install


# ---- discovery ----


def test_clean_run_has_no_warnings(search):
    search(FakeSearch())
    companies, warnings = discovery.run_discovery(FakeLLM(), REQUEST)
    assert companies
    assert warnings == []


def test_all_searches_failing_is_reported_as_unavailable_not_as_no_competitors(search):
    search(FakeSearch(fail=lambda q: True))
    companies, warnings = discovery.run_discovery(FakeLLM(), REQUEST)
    assert companies == []
    assert codes(warnings) == ["search_unavailable"]
    w = warnings[0]
    assert w.severity == "error"
    assert (w.details.stage, w.details.attempted, w.details.failed) == ("discovery", 10, 10)
    assert w.details.error_types == {"ForbiddenError": 10}


def test_partial_discovery_failure_still_returns_results(search):
    failing = set(discovery.ROUND_1_QUERIES[:3])
    search(FakeSearch(fail=lambda q: q in failing))
    companies, warnings = discovery.run_discovery(FakeLLM(), REQUEST)
    assert len(companies) == len(discovery.ROUND_1_QUERIES) - 3
    assert codes(warnings) == ["partial_search_failure"]
    w = warnings[0]
    assert w.severity == "warning"
    assert (w.details.stage, w.details.attempted, w.details.failed) == ("discovery", 10, 3)


def test_company_with_no_evidence_is_left_unscored_not_vetoed(search):
    # Profile queries embed the company name; fail all three for one company.
    target = None

    def fail(query):
        return target is not None and query.startswith(target)

    search(FakeSearch(fail=fail))
    companies_clean, _ = discovery.run_discovery(FakeLLM(), REQUEST)
    target = companies_clean[0].canonical_name

    companies, warnings = discovery.run_discovery(FakeLLM(), REQUEST)
    assert target not in [c.canonical_name for c in companies]  # dropped, not scored "exclude"
    assert "evidence_unavailable" in codes(warnings)
    evidence = next(w for w in warnings if w.code == "evidence_unavailable")
    # the UI renders these from details, never by parsing the message
    assert [c.name for c in evidence.details.companies] == [target]
    assert evidence.details.companies[0].url == companies_clean[0].primary_url


def test_model_skipping_a_company_when_scoring_is_reported(search):
    search(FakeSearch())
    clean, _ = discovery.run_discovery(FakeLLM(), REQUEST)
    skipped = clean[0]
    companies, warnings = discovery.run_discovery(FakeLLM(skip_scoring={skipped.primary_url}), REQUEST)
    assert codes(warnings) == ["scoring_incomplete"]
    assert [c.name for c in warnings[0].details.companies] == [skipped.canonical_name]
    assert len(companies) == len(clean) - 1


# ---- analyze ----


class FakeAgentLLM:
    """Wants to search on every turn (so it runs to the cap) unless
    stop_after is set; complete_structured is only allowed when sources exist."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.stop_after = stop_after
        self.turns = 0
        self.synthesized = False

    def complete_with_tools(self, system_prompt, messages, tools):
        self.turns += 1
        if self.stop_after is not None and self.turns > self.stop_after:
            return SimpleNamespace(content=[], stop_reason="end_turn")
        block = ToolUseBlock(type="tool_use", id=f"t{self.turns}", name="search", input={"query": f"q{self.turns}"})
        return SimpleNamespace(content=[block], stop_reason="tool_use")

    def complete_structured(self, system_prompt, message, output_format, output_config=None):
        self.synthesized = True
        url = _allowed_urls(output_format, "products", "source_url")[0]
        return output_format.model_validate(
            {
                "competitor_name": "Acme",
                "summary": "s",
                "products": [{"claim": "c", "source_url": url}],
                "pricing_signals": [],
                "regional_presence": [],
                "recommended_actions": [],
            }
        )


def test_no_sources_skips_synthesis_and_says_why(search):
    search(FakeSearch(fail=lambda q: True))
    llm = FakeAgentLLM(stop_after=2)
    report = agent.run_agentic_analysis(llm, AnalysisRequest(competitor_name="Acme"))
    assert llm.synthesized is False  # no LLM call, nothing to cite
    assert report.products == [] and report.regional_presence == []
    assert codes(report.warnings) == ["partial_search_failure", "no_sources"]
    assert report.warnings[1].severity == "error"
    assert report.warnings[1].details.stage == "research"


def test_iteration_cap_is_reported(search):
    search(FakeSearch())
    report = agent.run_agentic_analysis(FakeAgentLLM(), AnalysisRequest(competitor_name="Acme"))
    assert codes(report.warnings) == ["iteration_cap_hit"]
    assert report.products  # report still generated


def test_clean_analysis_has_no_warnings_and_keeps_report_shape(search):
    search(FakeSearch())
    report = agent.run_agentic_analysis(FakeAgentLLM(stop_after=1), AnalysisRequest(competitor_name="Acme"))
    assert report.warnings == []
    assert isinstance(report, AnalysisResponse)
    # warnings is code-only: it must never be part of the schema the model fills in
    assert "warnings" not in AnalysisResponse.model_json_schema()["properties"]


def test_search_tally_counts_by_error_type():
    tally = SearchTally()
    tally.record()
    tally.record(ForbiddenError("x"))
    tally.record(ForbiddenError("y"))
    assert (tally.attempted, tally.failed, tally.all_failed) == (3, 2, False)
    assert tally.describe_errors() == "ForbiddenError ×2"


def test_every_code_has_a_severity():
    # A new code added to WarningCode without a SEVERITY entry would crash
    # make_warning at runtime — catch it here instead.
    assert set(get_args(WarningCode)) == set(SEVERITY)
