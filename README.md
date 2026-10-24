# Competitor Intelligence Service

A FastAPI service that finds a company's competitors and researches them. Reports cite only sources retrieved during the run, enforced by the output schema.

Built for a B2B optical-equipment exporter, where competitor research was a manual, multi-hour task. It replaces an earlier prototype on a multi-agent framework (OpenClaw), where execution ran inside a third-party package and tracing and error handling could not be added.

## What it does

The service has two endpoints:

1. **`/discover`** -- given a description of your products, finds and scores candidate competitors.
2. **`/analyze`** -- given one competitor, researches it and writes a sourced report on products, pricing, and markets.

They are separate so that scoring, which is cheap, can screen candidates before a deep-dive, which is not.

`/discover` runs in four steps:

1. **Search** -- run 10 fixed queries, 7 in Chinese and 3 in English.
2. **Classify** -- label each result as manufacturer, trading company, overseas distributor, domestic-only seller, platform, media, or uncertain, and keep manufacturers, trading companies, and uncertain results.
3. **Merge** -- combine results that belong to the same company and resolve each company's homepage.
4. **Score** -- search each company's products, exports, and background, then rate product overlap, export orientation, and information availability from 1 to 10.

A run typically returns 15 to 18 companies.

`/analyze` runs a research loop of up to 5 rounds. The model chooses its own searches and stops when it has enough, then writes the report.

## Installation

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive API docs are served at `http://localhost:8000/docs`.

With Docker:

```bash
docker build -t competitor-analyst-agent .
docker run -p 8000:8000 --env-file .env competitor-analyst-agent
```

The image is a multi-stage build, runs as a non-root user, and runs uvicorn as PID 1 so it receives shutdown signals directly.

## Configuration

| Variable | Required | Purpose |
|------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes | LLM calls. The app does not start without it. |
| `TAVILY_API_KEYS` or `TAVILY_API_KEY` | Yes | Web search. `TAVILY_API_KEYS` takes a comma-separated list; a key that is out of quota, rate limited, or invalid hands over to the next. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | No | Tracing. Without them, tracing is disabled. |
| `LANGFUSE_BASE_URL` | No | Defaults to Langfuse Cloud. |

## Usage

```bash
curl -X POST localhost:8000/discover \
  -H "Content-Type: application/json" \
  -d '{"our_products": "Exporter of phoropters, auto refractometers, slit lamps and lensmeters"}'
```

Response, trimmed to one company:

```json
{
  "companies": [
    {
      "canonical_name": "Example Optics Co., Ltd.",
      "primary_url": "https://example-optics.example",
      "entity_type": "manufacturer",
      "product_overlap": 9,
      "export_orientation": 8,
      "info_availability": 8,
      "llm_recommendation": "deep_dive",
      "rule_recommendation": "deep_dive",
      "rule_score": 8.5,
      "agrees": true,
      "reasoning": "Manufactures auto refractometers and lensmeters; exports to 60+ countries."
    }
  ],
  "warnings": []
}
```

```bash
curl -X POST localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"competitor_name": "Example Optics Co., Ltd.", "competitor_url": "https://example-optics.example"}'
```

## Warnings

A failed search is skipped so that one failure does not end the run. Both endpoints return 200 with a `warnings` list describing anything that makes the result incomplete. The list is empty when the run completes cleanly.

```json
{
  "code": "search_unavailable",
  "severity": "error",
  "message": "All 10 discovery searches failed (ForbiddenError ×10). ...",
  "details": {"stage": "discovery", "attempted": 10, "failed": 10, "error_types": {"ForbiddenError": 10}}
}
```

`code`, `severity`, and `details` are for clients to build their own messages from. `message` is intended for developers and logs.

| Code | Severity | Meaning |
|------|---------|---------|
| `search_unavailable` | error | Every discovery search failed. An empty result means no data, not no competitors. |
| `no_sources` | error | The deep-dive retrieved nothing to cite, so no report was generated. |
| `partial_search_failure` | warning | Some searches failed. Results are based on the rest. |
| `evidence_unavailable` | warning | All profile searches failed for some companies, so they were not scored. Listed in `details.companies`. |
| `scoring_incomplete` | warning | The model skipped some companies when scoring. Listed in `details.companies`. |
| `classification_incomplete` | info | The model skipped some search results when classifying. |
| `iteration_cap_hit` | info | The deep-dive reached its 5-round limit while still searching. |

## Testing and evaluation

Unit tests cover the deterministic parts: the recommendation rule, URL handling, source constraints, key failover, trace propagation, and warnings. They need no API keys or network access and run in CI on every pull request.

```bash
pip install -r requirements-dev.txt
pytest
```

The discovery eval runs the real `/discover` against a hand-labeled list of known competitors. It reports recall, how many were ranked `deep_dive`, label accuracy, model/rule agreement, and duplicates.

```bash
python -m evals.run_discovery_eval            # one run
python -m evals.run_discovery_eval --runs 3   # run-to-run stability
```

Each run calls the real APIs: about $0.25 in LLM cost and about 130 Tavily credits. The competitor list is internal data and is gitignored. Create `evals/golden_set.json` in the format of [`evals/golden_set.example.json`](evals/golden_set.example.json).

Findings from the eval so far:

- When the search quota ran out, discovery returned an empty or partly unscored result with no indication of failure. This is now reported through warnings.
- The model labeled every known trading company as a manufacturer. Trading companies in this industry commonly present themselves as factories, so the label is not used in any recommendation.
- Recall varies between runs with no code changes, because live search results change.

## Cost and latency

Measured from Langfuse traces of real runs:

| Operation | Latency | LLM cost |
|------|---------|---------|
| Single structured call | ~11s | ~$0.01 |
| `/analyze` | 70–160s | $0.09–0.22 |
| `/discover` with scoring | ~90s | $0.15–0.30 |

Search is billed separately by Tavily; a `/discover` run makes about 65 searches. Deep-dive cost grows with each round because the full conversation is resent, so the round limit also caps cost.

## Design principles

- **Sources are constrained by schema.** Each run turns the URLs it retrieved into an enum in the output schema, so an unretrieved URL fails validation. This applies to every step that outputs a URL. It guarantees a cited source exists, not that it supports the claim.
- **The model rates, code decides.** The model scores each company, since judging product similarity needs language understanding. A fixed rule (`product×0.5 + export×0.3 + info×0.2`, excluding companies with almost no public information) makes the recommendation. Both are returned, and `agrees` marks where they differ.
- **Search where the companies are.** English queries mostly return B2B marketplaces and supplier lists. Chinese queries reach manufacturers' own sites and found more new companies per search on real runs.
- **Keep features that measure well.** A second search round for similar companies was built, measured, and removed after it found no companies the first round had missed.
- **Trace in one place.** Tracing is attached inside the LLM client, so every call nests under its request without tracing code at each call site.
- **Report partial results.** Failures that reduce the data behind a result are returned as warnings rather than hidden.

## Files

| File | Purpose |
|------|---------|
| `app/main.py` | Routes and HTTP status handling |
| `app/agent.py` | `/analyze` research loop |
| `app/discovery.py` | `/discover` pipeline |
| `app/llm.py` | Anthropic client and Langfuse tracing |
| `app/diagnostics.py` | Response warnings and search failure counts |
| `app/models.py`, `app/discovery_models.py` | Request and response schemas |
| `app/prompts.py`, `app/discovery_prompts.py` | Prompts |
| `app/tools/tavily_search.py` | Search client with key failover, and the search tool definition |
| `tests/` | Unit tests |
| `evals/` | Discovery eval and golden set template |

## Roadmap

- Remove the plain-text summary the research loop writes before the structured report.
- Retry or fall back to another model when the LLM API fails.
- Store past runs so repeat checks report only changes.
- Cache search results in evals so comparisons use fixed inputs.
- Merge companies that appear under both a Chinese and an English name.

## Stack

Python, FastAPI, Anthropic SDK, Pydantic, Tavily, Langfuse, OpenTelemetry, Docker
