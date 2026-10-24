"""The structural anti-fabrication guarantee: a source URL the model wasn't
shown this run must be an invalid value, not just a discouraged one."""

import pytest
from pydantic import ValidationError

from app.discovery_models import build_extraction_result_type, build_scoring_result_type
from app.models import AnalysisResponse, build_sourced_response_type
from app.prompts import build_synthesis_message

REAL_URLS = ["https://real-one.com/about", "https://real-two.com/products"]
FABRICATED_URL = "https://made-up.com/page"


def _report(source_url: str) -> dict:
    return {
        "competitor_name": "Acme",
        "summary": "s",
        "products": [{"claim": "makes widgets", "source_url": source_url}],
        "pricing_signals": [],
        "regional_presence": [],
        "recommended_actions": [],
    }


def test_retrieved_url_is_accepted():
    response_type = build_sourced_response_type(REAL_URLS)
    report = response_type.model_validate(_report(REAL_URLS[0]))
    assert report.products[0].source_url == REAL_URLS[0]


def test_fabricated_url_is_rejected():
    response_type = build_sourced_response_type(REAL_URLS)
    with pytest.raises(ValidationError):
        response_type.model_validate(_report(FABRICATED_URL))


def test_constraint_reaches_the_json_schema_sent_to_the_model():
    # Validation alone only catches it after the fact — the point is that the
    # schema the model generates against already has the enum in it.
    schema_text = str(build_sourced_response_type(REAL_URLS).model_json_schema())
    assert "'enum'" in schema_text
    for url in REAL_URLS:
        assert url in schema_text


def test_dynamic_type_still_serializes_as_the_base_response():
    report = build_sourced_response_type(REAL_URLS).model_validate(_report(REAL_URLS[0]))
    assert isinstance(report, AnalysisResponse)


def test_synthesis_refuses_to_run_without_sources():
    # Used to fall back to an unconstrained schema here. With zero URLs there
    # is no enum to build, so synthesis must not run at all (the agent
    # reports no_sources instead — see tests/test_warnings.py).
    with pytest.raises(ValueError):
        build_synthesis_message("Analyze: Acme", [])
    with pytest.raises(ValueError):
        build_synthesis_message("Analyze: Acme", [("q", "ForbiddenError: plan limit")])


def test_extraction_verdict_url_is_constrained_to_its_batch():
    result_type = build_extraction_result_type(REAL_URLS)
    verdict = {
        "company_name": "Acme",
        "is_own_page": True,
        "entity_type": "manufacturer",
        "evidence": "e",
    }
    result_type.model_validate({"verdicts": [{**verdict, "url": REAL_URLS[0]}]})
    with pytest.raises(ValidationError):
        result_type.model_validate({"verdicts": [{**verdict, "url": FABRICATED_URL}]})


def test_score_must_reference_a_company_in_the_batch_and_stay_in_range():
    result_type = build_scoring_result_type(REAL_URLS)
    score = {
        "company_url": REAL_URLS[0],
        "reasoning": "r",
        "product_overlap": 8,
        "export_orientation": 5,
        "info_availability": 7,
        "llm_recommendation": "deep_dive",
    }
    result_type.model_validate({"scores": [score]})
    with pytest.raises(ValidationError):
        result_type.model_validate({"scores": [{**score, "company_url": FABRICATED_URL}]})
    with pytest.raises(ValidationError):
        result_type.model_validate({"scores": [{**score, "product_overlap": 11}]})
