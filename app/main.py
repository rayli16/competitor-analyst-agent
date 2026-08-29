import logging

from fastapi import FastAPI, HTTPException
from langfuse import get_client, observe
from pydantic import ValidationError

from app.agent import run_agentic_analysis
from app.discovery import run_discovery
from app.discovery_models import DiscoveryRequest, DiscoveryResponse
from app.llm import LLMClient
from app.models import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Competitor Analysis Agent")

# Built at import time, not inside a route — if the API key is missing, we
# want the process to fail on startup, not on the first request that hits it.
llm_client = LLMClient()


# Liveness check — deliberately independent of the LLM call path.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
@observe(name="analyze", as_type="span")
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    try:
        return run_agentic_analysis(llm_client, request)
    except ValidationError as e:
        # Full exception (which field, why) goes server-side only — the
        # client gets a generic message, not internal schema details.
        logger.exception("complete_structured validation failed in /analyze")
        # complete_structured's gen.update() never runs on this path, so the
        # generation span itself stays blank — record the failure on the
        # currently-active span so the trace isn't silent about it.
        get_client().update_current_span(level="ERROR", status_message=str(e))
        # Upstream gave us something unusable — 502, not our bug (500).
        raise HTTPException(status_code=502, detail="Model returned data that didn't match the expected schema.")


@app.post("/discover")
@observe(name="discover", as_type="span")
def discover(request: DiscoveryRequest) -> DiscoveryResponse:
    try:
        companies = run_discovery(llm_client, request)
    except ValidationError as e:
        logger.exception("complete_structured validation failed in /discover")
        get_client().update_current_span(level="ERROR", status_message=str(e))
        raise HTTPException(status_code=502, detail="Model returned data that didn't match the expected schema.")
    return DiscoveryResponse(companies=companies)
