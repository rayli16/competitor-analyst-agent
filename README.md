# Competitor Intelligence Service

A FastAPI service that researches competitors on its own. You give it a company name and it runs a ReAct loop — deciding what to search, when it has enough, and writing up a sourced report. You give it nothing but your own product line and it goes the other direction: finds candidate competitors, scores them, and tells you which are worth a closer look.

I built it for a B2B optical-equipment exporter, where "who are our competitors" was a manual research job that ate hours. The interesting part isn't the LLM calls — it's the parts around them: making fabricated citations structurally impossible, keeping cost predictable with a discovery funnel, and knowing (from traces, not guesses) what every run actually costs.

## Why this exists

I first built this as a multi-agent system on an existing framework (OpenClaw). It worked, but the agents were configuration files — the execution engine lived inside a third-party package I didn't own. That meant no place to add real observability or fault handling, and nothing I could point to as engineering I'd actually done. So I pulled the core intelligence pipeline out and wrote it as a service I control end to end. This is that service.

## What it does

Two endpoints, two different jobs.

**`/analyze`** — deep research on one named competitor. A ReAct loop with web search: the model picks its own queries, reads results, and decides when it's seen enough (`stop_reason`, with an iteration cap as a backstop). It then writes a structured report where every product, price, and regional claim carries a source URL.

**`/discover`** — the other half of the problem. Identifying *who* your competitors are is the part a human does worst at scale, so this endpoint does it: ~10 hand-written search queries (mostly Chinese — see below), classify and dedupe the results with the LLM, resolve each company's homepage, then optionally score each one against your own product line on three dimensions.

Deep-diving is deliberately a separate call. `/discover` scores and ranks; you look at the scores and decide which companies are worth the `/analyze` spend. Keeping them separate is what makes the funnel save money (below).

## The parts I'd actually talk about

**Fabricated citations are structurally impossible, not just discouraged.** The obvious way to get sourced claims is to ask the model nicely to cite only real URLs. That's probabilistic — it'll usually comply. Instead, each run compiles the URLs it actually retrieved into a `Literal[tuple(urls)]` type, which becomes a real `enum` in the JSON schema sent to the model. A URL that wasn't retrieved this run isn't a discouraged answer — it's an invalid schema value, rejected at parse time. Verified by feeding it a fabricated URL and watching Pydantic reject it. Same trick is reused at every stage that emits a URL (classification, dedup, scoring, final report), so a claim can never point at a source that wasn't in front of the model.

To be precise about what this does and doesn't cover: it closes "cited a URL that was never retrieved." It does **not** close "the claim text itself is wrong" or "a real URL attached to the wrong claim" — those still depend on the model. The structural guarantee is about provenance, not truth.

**The discovery funnel keeps cost predictable.** Deep-diving every candidate would be the naive approach. Instead: discover ~13–15 candidates, run a cheap scoring pass, and only deep-dive the ones that clear a threshold. Cheap operations filter; the expensive one runs on a few. A scoring pass costs a fraction of one deep-dive, so screening first and deep-diving selectively costs far less than deep-diving everything blind.

**LLM judges, code decides.** Scoring splits the work: the model rates three dimensions (product overlap, export orientation, info availability) because that needs understanding — is "综合验光仪" the same category as "自动验光仪"? A rule can't tell. But turning scores into a recommendation is a fixed policy (`product*0.5 + export*0.3 + info*0.2`, with a hard veto if there's too little public info to research the company at all), so code does that. Both the model's own recommendation and the rule's are kept side by side with an `agrees` flag — the thresholds are policy you can tune in one place without touching the prompt.

**Chinese-first search, because the data said so.** English queries mostly surface B2B aggregators (Alibaba, Made-in-China, "top 100" listicles) — portals built for overseas buyers, not the manufacturers themselves. Chinese queries hit manufacturers' own sites directly; a lot of the smaller firms have little or no English presence at all. Measured on real runs, Chinese queries turned up noticeably more new manufacturer domains per search, so the query list is 7 Chinese to 3 English.

**Cuts made on evidence, not taste.** I planned a second "companies similar to X" search round to expand the candidate set. Built it, measured it — even after fixing the seed selection to only expand from confident manufacturer verdicts, it found zero companies round one hadn't already found. Cut it. The code comment records why, so I don't re-add it in six months.

## Cost and latency (from traces, not estimates)

Every LLM call is instrumented with Langfuse, so these are read off real runs:

| Operation | Latency | Cost |
|---|---|---|
| Single structured call (no tools) | ~11s | ~$0.01 |
| One `/analyze` deep-dive | ~70–160s | ~$0.09–0.22 |
| One `/discover` + scoring run | ~90s | ~$0.15–0.30 |

Deep-dive cost varies with how many search rounds the model runs, and later loop turns get more expensive because the whole conversation history is resent each turn — so the iteration cap is a cost control, not just a latency one.

## How observability is wired

Langfuse (v4, on OpenTelemetry) propagates trace context automatically through contextvars. That let me instrument in one place — inside the LLM client — and get every call correctly nested under its request, including every iteration of the agent loop, with zero tracing code at the call sites. The alternative (instrument at each call site) is coverage-by-discipline: add a call later, forget the tracing block, and it silently produces no traces. Structural coverage beats discipline.

Config failure is deliberately asymmetric: a missing `ANTHROPIC_API_KEY` or `TAVILY_API_KEY` crashes the process at startup (a service that looks alive but can only 500 is worse than one that won't boot), while missing Langfuse keys degrade quietly to a no-op tracer (a broken observability backend shouldn't take down a working endpoint).

## Running it

```bash
cp .env.example .env          # then fill in your keys
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`GET /health` returns `{"status": "ok"}` once it's up. `/analyze` and `/discover` are POST — see the auto-generated docs at `/docs`.

Requires `ANTHROPIC_API_KEY` and a Tavily key (both fail-fast if missing). Tavily takes either `TAVILY_API_KEY` or a comma-separated `TAVILY_API_KEYS` list: keys are tried in order, and a key that is out of quota, rate limited, or invalid hands over to the next one. Langfuse keys are optional — without them tracing just no-ops.

### Docker

```bash
docker build -t competitor-analyst-agent .
docker run -p 8000:8000 --env-file .env competitor-analyst-agent
```

Multi-stage build (build tooling never reaches the runtime image), Debian-slim rather than Alpine (the dependency list is compiled-extension-heavy — Alpine's musl libc can't use manylinux wheels), runs as a non-root user, and `exec`s uvicorn so it's PID 1 and receives SIGTERM directly for clean shutdown.

## Testing and evals

Two layers, split by what they cost.

**Unit tests** cover the deterministic parts: the recommendation rule and its thresholds, the coverage checks, URL extraction and homepage resolution, Tavily key failover, and the anti-fabrication guarantee itself (a URL that wasn't retrieved is rejected, and the constraint really does reach the JSON schema the model sees). No API keys, no network, under a second. CI runs them on every pull request.

```bash
pip install -r requirements-dev.txt
pytest
```

**The discovery eval** runs the real `/discover` pipeline against a hand-labeled golden set of known competitors and reports recall, how many of them landed in `deep_dive`, label accuracy, model/rule agreement, and duplicates (one real company surfacing as several entries). It calls the real APIs, so it's run by hand, not in CI.

```bash
python -m evals.run_discovery_eval            # one run
python -m evals.run_discovery_eval --runs 3   # run-to-run stability
```

The golden set is internal business data (who we consider competitors, and where we sell), so `evals/golden_set.json` is gitignored. Create it locally in the format of [`evals/golden_set.example.json`](evals/golden_set.example.json). Results are written to `evals/results/`, also gitignored.

What the eval has caught so far:

- **A silent failure.** When the Tavily quota ran out, discovery returned 0 companies instead of an error, because each search failure is skipped by design. Worse, when the quota ran out *mid*-run, the remaining companies were scored with no evidence and vetoed to `exclude`, a plausible-looking but wrong result. The eval now flags both cases.
- **An unreliable label.** The model classified every known trading company as a `manufacturer`, on every run. In this industry, trading companies routinely describe themselves as factories, so the distinction isn't observable from public pages. No recommendation depends on it.
- **Recall moves without code changes.** Live search results drift between runs, so a change only counts as an improvement if it beats that noise.

## Layout

```
app/
├── __init__.py
├── main.py                 # HTTP boundary — routes, validation, status codes
├── agent.py                # ReAct loop: orchestration, termination, tool dispatch
├── discovery.py            # the discovery/scoring funnel
├── llm.py                  # Anthropic client wrapper + Langfuse instrumentation
├── models.py               # request/response schemas for /analyze
├── discovery_models.py     # schemas for /discover
├── prompts.py              # prompts for /analyze
├── discovery_prompts.py    # prompts for /discover
└── tools/
    ├── __init__.py
    └── tavily_search.py    # Tavily search client (with key failover) + the search tool definition
tests/                      # unit tests — no network, run in CI
evals/
├── run_discovery_eval.py   # real-API discovery eval against the golden set
├── metrics.py              # recall/duplicate/stability scoring (unit tested)
└── golden_set.example.json # format only — the real golden set stays local
```

Each layer knows the one below it, never the one above — swapping `/analyze`'s engine from a single call to the full ReAct loop touched one line of `main.py`.

## What I'd improve next

The service works fine right now, and here is my plan on areas that I'd refine next:

- **Make `export_orientation` more precise.** It currently scores evidence of serving overseas markets. Splitting it into two signals, whether the company makes the product and whether it already exports, would separate potential competitors from overseas resellers more cleanly. This is also where the model's recommendation and the rule's recommendation differ most often, so it's the most useful place to refine.
- **Trim the final step of the deep-dive.** When the research loop finishes, the model's last turn often writes a plain-text summary before the structured report is generated. Adjusting the research prompt so the loop simply signals it's done would lower the cost per run.
- **Add retries and a fallback model.** Search failures inside the loop are already handled: the error goes back to the model as `is_error`, so it can try another query. Adding the same resilience at the LLM API level (retrying on timeouts, or switching to a backup model) would make long runs more reliable.
- **Add memory between runs.** Saving past results would let a repeat analysis reuse earlier research and highlight what changed since last time, which fits a weekly monitoring use case.
- **Make search failures visible to the caller.** Skipping a failed search keeps one bad query from sinking a run, but when most searches fail, the response should say the data is incomplete rather than return a short list that looks complete.
- **Make the eval reproducible.** Caching search results during evals would hold the inputs fixed, so before/after comparisons measure the code change alone, and reruns wouldn't spend search quota.

## Stack

Python · FastAPI · Anthropic SDK · Pydantic · Tavily · Langfuse / OpenTelemetry · Docker
