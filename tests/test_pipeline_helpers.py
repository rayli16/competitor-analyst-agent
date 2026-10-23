import pytest
from anthropic.types import ToolUseBlock

from app import discovery
from app.agent import extract_findings
from app.discovery import (
    check_full_coverage,
    check_full_scoring_coverage,
    deduplicate_by_url,
    derive_domain_roots,
    filter_competitors,
    resolve_primary_urls,
)
from app.discovery_models import CandidateScore, CandidateVerdict, Company, ResolvedCompany
from app.prompts import extract_source_urls


# ---- URL extraction (feeds the source enum) ----


def test_extract_source_urls_reads_only_title_lines():
    findings = [
        (
            "query",
            "Acme Optics (https://acme.com/about)\nWe partner with (https://quoted-in-body.com) and others.\n\n"
            "Second Result (https://second.com)\nbody text",
        )
    ]
    assert extract_source_urls(findings) == ["https://acme.com/about", "https://second.com"]


def test_extract_source_urls_dedupes_across_searches_in_order():
    findings = [
        ("q1", "A (https://a.com)\nx\n\nB (https://b.com)\ny"),
        ("q2", "A again (https://a.com)\nz"),
    ]
    assert extract_source_urls(findings) == ["https://a.com", "https://b.com"]


def test_extract_findings_pairs_results_by_tool_use_id_not_position():
    messages = [
        {"role": "user", "content": "Analyze: Acme"},
        {
            "role": "assistant",
            "content": [
                ToolUseBlock(type="tool_use", id="t1", name="search", input={"query": "first"}),
                ToolUseBlock(type="tool_use", id="t2", name="search", input={"query": "second"}),
            ],
        },
        {
            "role": "user",
            "content": [  # results arrive in the opposite order to the calls
                {"type": "tool_result", "tool_use_id": "t2", "content": "result-2"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "result-1"},
            ],
        },
    ]
    assert extract_findings(messages) == [("second", "result-2"), ("first", "result-1")]


# ---- dedup / coverage ----


def test_deduplicate_by_url_keeps_first_occurrence_order():
    results = [{"url": "u1", "n": 1}, {"url": "u2", "n": 2}, {"url": "u1", "n": 3}]
    assert deduplicate_by_url(results) == [{"url": "u1", "n": 1}, {"url": "u2", "n": 2}]


def _verdict(url: str, entity_type: str = "manufacturer") -> CandidateVerdict:
    return CandidateVerdict(company_name="c", url=url, is_own_page=True, entity_type=entity_type, evidence="e")


def test_coverage_check_catches_a_duplicate_hiding_a_skip():
    # Same count in and out, so a length check would pass — the set check must not.
    results = [{"url": "u1"}, {"url": "u2"}]
    verdicts = [_verdict("u1"), _verdict("u1")]
    assert check_full_coverage(results, verdicts) == {"u2"}


def test_coverage_check_passes_when_every_url_is_covered():
    assert check_full_coverage([{"url": "u1"}], [_verdict("u1")]) == set()


def test_scoring_coverage_check_reports_unscored_company():
    companies = [
        ResolvedCompany(
            canonical_name=n, urls=[u], entity_type="manufacturer", merge_reasoning="m", primary_url=u
        )
        for n, u in [("A", "https://a.com"), ("B", "https://b.com")]
    ]
    scores = [
        CandidateScore(
            company_url="https://a.com",
            reasoning="r",
            product_overlap=5,
            export_orientation=5,
            info_availability=5,
            llm_recommendation="monitor",
        )
    ]
    assert check_full_scoring_coverage(companies, scores) == {"https://b.com"}


def test_filter_competitors_keeps_uncertain_and_drops_non_competitors():
    verdicts = [
        _verdict("m", "manufacturer"),
        _verdict("t", "trading_company"),
        _verdict("u", "uncertain"),
        _verdict("d", "distributor"),
        _verdict("p", "platform"),
        _verdict("media", "media_or_event"),
        _verdict("ds", "domestic_seller"),
    ]
    assert [v.url for v in filter_competitors(verdicts)] == ["m", "t", "u"]


# ---- homepage resolution ----


def test_derive_domain_roots_collapses_deep_links_per_domain():
    urls = [
        "https://www.acme.com/products/a",
        "https://www.acme.com/about",
        "http://acme-vision.com/x",
    ]
    assert derive_domain_roots(urls) == ["https://www.acme.com", "http://acme-vision.com"]


def test_single_domain_company_resolves_without_a_search(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("single-domain companies should not trigger a homepage search")

    monkeypatch.setattr(discovery, "find_homepage_via_search", fail_if_called)
    company = Company(
        canonical_name="Acme",
        urls=["https://www.acme.com/products/a", "https://www.acme.com/about"],
        entity_type="manufacturer",
        merge_reasoning="m",
    )
    [resolved] = resolve_primary_urls([company])
    assert resolved.primary_url == "https://www.acme.com"


def test_multi_domain_company_resolves_via_search(monkeypatch):
    calls = []

    def fake_search(company_name, known_roots):
        calls.append((company_name, known_roots))
        return known_roots[1]

    monkeypatch.setattr(discovery, "find_homepage_via_search", fake_search)
    company = Company(
        canonical_name="Acme",
        urls=["https://acme-a.com/x", "https://acme-b.com/y"],
        entity_type="manufacturer",
        merge_reasoning="m",
    )
    [resolved] = resolve_primary_urls([company])
    assert calls == [("Acme", ["https://acme-a.com", "https://acme-b.com"])]
    assert resolved.primary_url == "https://acme-b.com"
