import pytest

from app.discovery import compute_recommendation, merge_scores
from app.discovery_models import CandidateScore, ResolvedCompany


# Inputs picked so the weighted sum is exact in floating point — e.g.
# 6*0.5 + 10*0.3 + 5*0.2 is exactly 7.0 — so the boundary cases really test
# the >= comparison, not float rounding.
@pytest.mark.parametrize(
    ("product", "export", "info", "expected", "expected_score"),
    [
        (6, 10, 5, "deep_dive", 7.0),  # exactly on the deep_dive threshold
        (10, 5, 5, "deep_dive", 7.5),
        (5, 5, 5, "monitor", 5.0),  # exactly on the monitor threshold
        (4, 5, 5, "exclude", 4.5),
    ],
)
def test_weighted_thresholds(product, export, info, expected, expected_score):
    recommendation, score = compute_recommendation(product, export, info)
    assert recommendation == expected
    assert score == pytest.approx(expected_score)


def test_low_info_is_vetoed_regardless_of_other_scores():
    assert compute_recommendation(10, 10, 3) == ("exclude", 0.0)


def test_veto_boundary_info_4_is_not_vetoed():
    recommendation, score = compute_recommendation(10, 10, 4)
    assert recommendation == "deep_dive"
    assert score == pytest.approx(8.8)


def _company(url: str) -> ResolvedCompany:
    return ResolvedCompany(
        canonical_name=url,
        urls=[url],
        entity_type="manufacturer",
        merge_reasoning="single candidate, no merge",
        primary_url=url,
    )


def _score(url: str, product: int, export: int, info: int, llm_recommendation: str) -> CandidateScore:
    return CandidateScore(
        company_url=url,
        reasoning="test",
        product_overlap=product,
        export_orientation=export,
        info_availability=info,
        llm_recommendation=llm_recommendation,
    )


def test_merge_scores_computes_rule_and_agreement():
    companies = [_company("https://a.com"), _company("https://b.com")]
    scores = [
        _score("https://a.com", 10, 5, 5, "deep_dive"),  # rule: deep_dive -> agrees
        _score("https://b.com", 5, 5, 5, "deep_dive"),  # rule: monitor -> disagrees
    ]
    merged = {c.primary_url: c for c in merge_scores(companies, scores)}
    assert merged["https://a.com"].rule_recommendation == "deep_dive"
    assert merged["https://a.com"].agrees is True
    assert merged["https://b.com"].rule_recommendation == "monitor"
    assert merged["https://b.com"].agrees is False


def test_merge_scores_drops_unscored_company_instead_of_faking_a_score():
    companies = [_company("https://a.com"), _company("https://skipped.com")]
    scores = [_score("https://a.com", 10, 5, 5, "deep_dive")]
    merged = merge_scores(companies, scores)
    assert [c.primary_url for c in merged] == ["https://a.com"]


def test_merge_scores_matches_by_url_not_list_position():
    companies = [_company("https://a.com"), _company("https://b.com")]
    scores = [  # deliberately reversed relative to companies
        _score("https://b.com", 1, 1, 5, "exclude"),
        _score("https://a.com", 10, 10, 10, "deep_dive"),
    ]
    merged = {c.primary_url: c for c in merge_scores(companies, scores)}
    assert merged["https://a.com"].product_overlap == 10
    assert merged["https://b.com"].product_overlap == 1
