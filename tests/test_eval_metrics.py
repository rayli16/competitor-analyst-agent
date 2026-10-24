import pytest

from app.discovery_models import ScoredCompany
from app.diagnostics import make_warning
from evals.metrics import aggregate_runs, is_valid_run, load_golden, match_reason, normalize_domain, score_run


def _scored(name: str, urls: list[str], rule: str = "deep_dive", llm: str = "deep_dive", score: float = 8.0):
    return ScoredCompany(
        canonical_name=name,
        urls=urls,
        entity_type="manufacturer",
        merge_reasoning="m",
        primary_url=urls[0],
        reasoning="r",
        product_overlap=8,
        export_orientation=8,
        info_availability=8,
        llm_recommendation=llm,
        rule_recommendation=rule,
        rule_score=score,
        agrees=llm == rule,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.Acme.com/about", "acme.com"),
        ("http://acme.com:8080", "acme.com"),
        ("www.acme.com", "acme.com"),
        ("acme.com", "acme.com"),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


def test_match_by_domain_including_subdomains():
    entry = {"name": "Acme", "domains": ["acme.com"]}
    assert match_reason(entry, _scored("Something Else", ["https://shop.acme.com/x"])) == "domain"


def test_domain_match_does_not_accept_lookalike_suffix():
    entry = {"name": "Zzz", "domains": ["acme.com"]}
    assert match_reason(entry, _scored("Other", ["https://notacme.com"])) is None


def test_match_by_alias_case_insensitive():
    entry = {"name": "Brightstar", "aliases": ["示例星光"], "domains": []}
    assert match_reason(entry, _scored("示例星光光学有限公司", ["https://shixing.example"])) == "name"
    assert match_reason(entry, _scored("EXAMPLE BRIGHTSTAR Optical", ["https://x.com"])) == "name"


def test_score_run_recall_duplicates_and_unmatched():
    golden = [
        {"name": "Brightstar", "aliases": ["示例星光"], "domains": [], "expected": "deep_dive"},
        {"name": "Acme", "domains": ["acme.com"], "expected": "monitor"},
        {"name": "Never Found", "domains": ["nowhere.com"]},
    ]
    companies = [
        _scored("示例星光光学有限公司", ["https://shixing.example"], rule="deep_dive", score=7.8),
        _scored("Example Brightstar Optical", ["https://brightstar.example"], rule="deep_dive", score=8.3),
        _scored("Acme Optics", ["https://www.acme.com"], rule="deep_dive", llm="monitor"),
        _scored("Random Noise Co", ["https://noise.com"], rule="exclude", llm="exclude", score=2.0),
    ]
    run = score_run(golden, companies)
    assert run["recall"] == pytest.approx(2 / 3)
    assert run["missed"] == ["Never Found"]
    assert run["duplicate_count"] == 1
    assert run["duplicates"][0]["golden"] == "Brightstar"
    # the higher-scoring duplicate represents the golden entry
    assert run["matched"][0]["found_as"] == "Example Brightstar Optical"
    assert run["unmatched_found"] == ["Random Noise Co"]
    # Brightstar labeled deep_dive -> correct; Acme labeled monitor but got deep_dive -> wrong
    assert run["label_accuracy"] == pytest.approx(0.5)
    assert run["deep_dive_recall"] == pytest.approx(2 / 3)
    assert run["agreement_rate"] == pytest.approx(3 / 4)


def test_aggregate_runs_tracks_score_spread():
    golden = [{"name": "Acme", "domains": ["acme.com"]}]
    runs = [
        score_run(golden, [_scored("Acme", ["https://acme.com"], score=7.0)]),
        score_run(golden, [_scored("Acme", ["https://acme.com"], score=8.5)]),
        score_run(golden, [_scored("Other", ["https://other.com"])]),
    ]
    summary = aggregate_runs(runs)
    assert summary["recall"]["min"] == 0
    assert summary["recall"]["max"] == 1
    assert summary["score_stability"]["Acme"] == {"found_in_runs": 2, "min": 7.0, "max": 8.5}


def test_empty_run_is_invalid_and_excluded_from_summary():
    golden = [{"name": "Acme", "domains": ["acme.com"]}]
    good = score_run(golden, [_scored("Acme", ["https://acme.com"])])
    empty = score_run(golden, [])
    summary = aggregate_runs([good, empty])
    assert summary["runs"] == 1
    assert summary["invalid_runs"] == 1
    assert summary["recall"]["min"] == 1  # not dragged to 0 by the failed run


def test_search_unavailable_warning_invalidates_run_and_is_recorded():
    golden = [{"name": "Acme", "domains": ["acme.com"]}]
    down = make_warning("search_unavailable", "all failed", stage="discovery", attempted=10, failed=10)
    run = score_run(golden, [], [down])
    assert is_valid_run(run) is False
    assert run["warnings"][0]["code"] == "search_unavailable"
    assert run["warnings"][0]["details"]["failed"] == 10


def test_partial_failure_warning_keeps_run_valid():
    golden = [{"name": "Acme", "domains": ["acme.com"]}]
    partial = make_warning("partial_search_failure", "3 of 10 failed")
    run = score_run(golden, [_scored("Acme", ["https://acme.com"])], [partial])
    assert is_valid_run(run) is True


def test_load_golden_rejects_empty_and_bad_labels():
    with pytest.raises(ValueError, match="no competitors"):
        load_golden({"our_products": "x", "competitors": []})
    with pytest.raises(ValueError, match="invalid 'expected'"):
        load_golden({"our_products": "x", "competitors": [{"name": "A", "expected": "maybe"}]})
