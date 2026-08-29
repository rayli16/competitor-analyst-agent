import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langfuse import get_client

from app.discovery_models import (
    CandidateScore,
    Company,
    CandidateVerdict,
    DiscoveryRequest,
    Recommendation,
    ResolvedCompany,
    ScoredCompany,
    build_dedup_result_type,
    build_extraction_result_type,
    build_scoring_result_type,
)
from app.discovery_prompts import (
    DEDUP_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    SCORING_SYSTEM_PROMPT,
    build_dedup_message,
    build_extraction_message,
    build_scoring_message,
)
from app.llm import LLMClient
from app.tools.tavily_search import TavilySearchClient

tavily_client = TavilySearchClient()

CONTENT_TRUNCATE_CHARS = 500  # extraction needs enough signal to classify, not the full page

# Hit max_tokens mid-JSON once at 47 results in a single call (a real
# ValidationError, not a hypothetical). 20 keeps each call's output well
# under the ceiling — not a tuned number, just a safe one.
EXTRACTION_BATCH_SIZE = 20

# This is classification, not open-ended synthesis, so it doesn't need the
# SDK's full adaptive-thinking default. Went with "medium" over "low" since
# entity_type has real ambiguity (see the candidate model) — worth tuning
# down once there's Langfuse data to justify it.
DISCOVERY_OUTPUT_CONFIG = {"effort": "medium"}

# Hand-picked domain knowledge, update as our product/geography scope
# changes. Manufacturer + trading-company angles, Chinese + English.
ROUND_1_QUERIES = [
    "视光设备 生产厂家",
    "验光设备 厂家 出口",
    "验光设备 厂家 中国",

    "综合验光仪 自动验光仪 电脑验光仪 厂家",
    "焦度计 磨边机 生产厂家",
    "裂隙灯 生物测量仪 眼科检查设备 厂家",
    "试镜片箱 视力表 瞳距仪 生产厂家",
    
    "optometric equipment manufacturer China export",
    "autorefractor phoropter lensmeter supplier China OEM",
    "ophthalmic equipment alibaba supplier China price",
]

# Same idea as ROUND_1_QUERIES — one company name fills all three. Covers
# product line, export orientation, and company background: the three
# things scoring needs evidence for.
PROFILE_QUERY_TEMPLATES = [
    "{company_name} 主营产品 产品线",
    "{company_name} 出口 海外市场 经销商",
    "{company_name} 公司简介 规模 成立",
]

# Batched for the same reason extraction is — max_tokens truncation already
# bit us once. Haven't actually measured this one though: per-company input
# is heavier than extraction's (up to 9 profile results vs. 1), but output
# is lighter (a few short fields vs. a full CandidateVerdict). Not obvious
# batching is even necessary here — just playing it safe.
SCORING_BATCH_SIZE = 8

# Capped rather than scaling with company count. Round 1's ~11-way
# concurrency has run fine in production, but profile search can hit 45+
# tasks (15 companies x 3 queries) — 4x that, and untested against Tavily's
# actual tolerance. Rather find out the safe limit than the hard way.
PROFILE_SEARCH_CONCURRENCY = 15

_seen_domains_lock = threading.Lock()


def dispatch_search(query: str, seen_domains: set[str]) -> list[dict[str, str]]:
    with get_client().start_as_current_observation(
        name="search",
        as_type="tool",
        input={"query": query},
    ) as tool_span:
        try:
            results = tavily_client.search(query)
        except Exception as e:
            # No LLM is watching this search individually (unlike the ReAct
            # loop's dispatch()) — log and skip, don't abort the whole run.
            tool_span.update(output=str(e), level="ERROR", status_message=str(e))
            return []
        # seen_domains is shared across concurrently-running searches in the
        # same round — lock the check-and-update so counts stay accurate.
        with _seen_domains_lock:
            new_domain_count = sum(1 for r in results if urlparse(r["url"]).netloc not in seen_domains)
            seen_domains.update(urlparse(r["url"]).netloc for r in results)
        tool_span.update(
            output=[r["url"] for r in results],
            metadata={"result_count": len(results), "new_domain_count": new_domain_count},
        )
        return results


def run_search_round(queries: list[str], seen_domains: set[str]) -> list[dict[str, str]]:
    if not queries:
        # ROUND_1_QUERIES is a maintained, hand-edited constant — guard the
        # accidentally-empty case rather than let ThreadPoolExecutor(max_workers=0)
        # raise ValueError.
        return []
    # Queries within a round are independent — run concurrently. I/O-bound
    # (waiting on Tavily), so a thread pool is a good fit; search() itself
    # is synchronous, not async.
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        batches = pool.map(lambda q: dispatch_search(q, seen_domains), queries)
    return [result for batch in batches for result in batch]


def deduplicate_by_url(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduplicated: list[dict[str, str]] = []
    for result in results:
        if result["url"] in seen:
            continue
        seen.add(result["url"])
        deduplicated.append(result)
    return deduplicated


def check_full_coverage(input_results: list[dict[str, str]], verdicts: list[CandidateVerdict]) -> set[str]:
    # Stronger than a length check: catches the model covering one URL twice
    # while skipping another, which a bare len(output) == len(input) wouldn't.
    input_urls = {r["url"] for r in input_results}
    output_urls = {v.url for v in verdicts}
    return input_urls - output_urls


def classify_batch(llm_client: LLMClient, batch: list[dict[str, str]]) -> list[CandidateVerdict]:
    # response_type is built from THIS batch's URLs only — not the full
    # result set. Constraining the enum to all 47 while only 20 are in the
    # prompt would let the model "correctly" cite a URL it was never shown.
    response_type = build_extraction_result_type([r["url"] for r in batch])
    message = build_extraction_message(batch)
    result = llm_client.complete_structured(
        EXTRACTION_SYSTEM_PROMPT, message, response_type, output_config=DISCOVERY_OUTPUT_CONFIG
    )
    return result.verdicts


def extract_candidates(
    llm_client: LLMClient, results: list[dict[str, str]]
) -> tuple[list[CandidateVerdict], set[str]]:
    truncated = [{**r, "content": r["content"][:CONTENT_TRUNCATE_CHARS]} for r in results]
    batches = [truncated[i : i + EXTRACTION_BATCH_SIZE] for i in range(0, len(truncated), EXTRACTION_BATCH_SIZE)]
    if not batches:
        return [], set()
    # Batches are independent — same concurrency pattern as run_search_round.
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        batch_verdicts = pool.map(lambda batch: classify_batch(llm_client, batch), batches)
    verdicts = [v for verdicts in batch_verdicts for v in verdicts]
    # Checked against the FULL result set, after merging — not per-batch.
    # A per-batch check would only ever catch a batch skipping its own
    # results, not the merged picture check_full_coverage is meant to give.
    missing = check_full_coverage(truncated, verdicts)
    return verdicts, missing


def gather_and_classify_candidates(llm_client: LLMClient) -> tuple[list[CandidateVerdict], set[str]]:
    # Seed expansion (a second, "companies similar to X" search round) was
    # tried and cut: measured on a real run, after fixing seed selection to
    # only expand from confident manufacturer/trading_company verdicts, it
    # still contributed zero companies round 1 wouldn't have found anyway.
    seen_domains: set[str] = set()
    raw_results = deduplicate_by_url(run_search_round(ROUND_1_QUERIES, seen_domains))
    if not raw_results:
        return [], set()
    return extract_candidates(llm_client, raw_results)


COMPETITOR_ENTITY_TYPES = {"manufacturer", "trading_company", "uncertain"}


def filter_competitors(verdicts: list[CandidateVerdict]) -> list[CandidateVerdict]:
    # uncertain is included deliberately (see the candidate model) — an
    # unclear verdict gets surfaced for review, not silently discarded as noise.
    return [v for v in verdicts if v.entity_type in COMPETITOR_ENTITY_TYPES]


def deduplicate_candidates(llm_client: LLMClient, verdicts: list[CandidateVerdict]) -> list[Company]:
    candidates = filter_competitors(verdicts)
    if not candidates:
        return []
    response_type = build_dedup_result_type([c.url for c in candidates])
    message = build_dedup_message(candidates)
    result = llm_client.complete_structured(
        DEDUP_SYSTEM_PROMPT, message, response_type, output_config=DISCOVERY_OUTPUT_CONFIG
    )
    return result.companies


def derive_domain_roots(urls: list[str]) -> list[str]:
    # Unique domain roots, in original order. For a company with one known
    # domain, the root is almost always the real homepage — deep/product/
    # download-page URLs all collapse down to it anyway.
    seen_domains: set[str] = set()
    roots: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc in seen_domains:
            continue
        seen_domains.add(parsed.netloc)
        roots.append(f"{parsed.scheme}://{parsed.netloc}")
    return roots


def find_homepage_via_search(company_name: str, known_roots: list[str]) -> str:
    # Only reached for genuinely ambiguous, multi-domain companies (see
    # resolve_primary_urls) — the minority case, not the default path.
    with get_client().start_as_current_observation(
        name="find_homepage", as_type="tool", input={"company_name": company_name}
    ) as tool_span:
        results = tavily_client.search(f"{company_name} official website", max_results=3)
        known_domains = {urlparse(root).netloc for root in known_roots}
        for result in results:
            domain = urlparse(result["url"]).netloc
            if domain in known_domains:
                # The search confirmed one of the domains we already knew
                # about — trust it over an arbitrary pick among the roots.
                tool_span.update(output=result["url"], metadata={"confirmed": True})
                return f"{urlparse(result['url']).scheme}://{domain}"
        # Search didn't confirm any already-known domain — don't trust an
        # unconfirmed new one; fall back to the first known root instead.
        tool_span.update(output=known_roots[0], metadata={"confirmed": False})
        return known_roots[0]


def resolve_primary_urls(companies: list[Company]) -> list[ResolvedCompany]:
    resolved: list[ResolvedCompany] = []
    for company in companies:
        roots = derive_domain_roots(company.urls)
        # One known domain: free, no search needed. Multiple domains for one
        # company (the real six-domain case this whole pipeline is built
        # around) is exactly the ambiguity the extra search resolves.
        primary_url = roots[0] if len(roots) == 1 else find_homepage_via_search(company.canonical_name, roots)
        resolved.append(ResolvedCompany(**company.model_dump(), primary_url=primary_url))
    return resolved


def build_profile_queries(company_name: str) -> list[str]:
    return [template.format(company_name=company_name) for template in PROFILE_QUERY_TEMPLATES]


def dispatch_profile_search(company_url: str, query: str) -> list[dict[str, str]]:
    # A separate function from dispatch_search, not a reuse with a throwaway
    # seen_domains — new_domain_count means "found a company we haven't seen
    # yet," which isn't a meaningful concept here: profile search gathers
    # evidence on companies we already have, it doesn't discover new ones.
    with get_client().start_as_current_observation(
        name="profile_search", as_type="tool", input={"company_url": company_url, "query": query}
    ) as tool_span:
        try:
            # Smaller than discovery search's default of 5 — a targeted
            # single-company query needs fewer results than a broad category
            # search, and it keeps the scoring input smaller too.
            results = tavily_client.search(query, max_results=3)
        except Exception as e:
            tool_span.update(output=str(e), level="ERROR", status_message=str(e))
            return []
        tool_span.update(output=[r["url"] for r in results], metadata={"result_count": len(results)})
        return results


def gather_company_profiles(companies: list[ResolvedCompany]) -> dict[str, list[dict[str, str]]]:
    if not companies:
        return {}
    # One flat pool of every (company, query) task instead of batching per
    # company — per-company batching would mean company 15 doesn't start
    # until companies 1-14 have each finished all 3 of their queries. A flat
    # pool lets everything overlap regardless of which company a task
    # belongs to. Capped at PROFILE_SEARCH_CONCURRENCY, not left uncapped —
    # see that constant's comment.
    tasks = [
        (company.primary_url, query) for company in companies for query in build_profile_queries(company.canonical_name)
    ]
    with ThreadPoolExecutor(max_workers=min(len(tasks), PROFILE_SEARCH_CONCURRENCY)) as pool:
        batches = pool.map(lambda task: dispatch_profile_search(*task), tasks)
    # Grouped back by company_url here — dispatch ran as a flat pool, but
    # the scoring call needs each company's evidence kept separate.
    profiles: dict[str, list[dict[str, str]]] = {company.primary_url: [] for company in companies}
    for (company_url, _query), batch in zip(tasks, batches):
        profiles[company_url].extend(batch)
    return profiles


def check_full_scoring_coverage(companies: list[ResolvedCompany], scores: list[CandidateScore]) -> set[str]:
    # Same set-based check as check_full_coverage in extraction, same reason:
    # catches the model scoring one company twice while skipping another.
    company_urls = {c.primary_url for c in companies}
    scored_urls = {s.company_url for s in scores}
    return company_urls - scored_urls


def score_batch(
    llm_client: LLMClient,
    companies: list[ResolvedCompany],
    profiles: dict[str, list[dict[str, str]]],
    our_products: str,
) -> list[CandidateScore]:
    # response_type constrained to THIS batch's company URLs only — same
    # reason as build_verdict_type: a score can't correlate to a company
    # that wasn't actually in this call.
    response_type = build_scoring_result_type([c.primary_url for c in companies])
    message = build_scoring_message(companies, profiles, our_products, CONTENT_TRUNCATE_CHARS)
    result = llm_client.complete_structured(
        SCORING_SYSTEM_PROMPT, message, response_type, output_config=DISCOVERY_OUTPUT_CONFIG
    )
    return result.scores


def score_companies(
    llm_client: LLMClient,
    companies: list[ResolvedCompany],
    profiles: dict[str, list[dict[str, str]]],
    our_products: str,
) -> tuple[list[CandidateScore], set[str]]:
    if not companies:
        return [], set()
    batches = [companies[i : i + SCORING_BATCH_SIZE] for i in range(0, len(companies), SCORING_BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        batch_scores = pool.map(lambda batch: score_batch(llm_client, batch, profiles, our_products), batches)
    scores = [s for scores in batch_scores for s in scores]
    missing = check_full_scoring_coverage(companies, scores)
    return scores, missing


def compute_recommendation(product: int, export: int, info: int) -> tuple[Recommendation, float]:
    # LLM judges (needs understanding), code decides (needs consistency).
    # Thresholds are policy — adjustable in one place without touching the
    # scoring prompt/schema.
    if info <= 3:
        # Hard veto: we've seen deep-dive quality collapse without enough
        # public info to work with — no amount of product overlap justifies
        # spending $0.216 on a company we can't actually research.
        return "exclude", 0.0
    overall = product * 0.5 + export * 0.3 + info * 0.2
    if overall >= 7:
        return "deep_dive", overall
    if overall >= 5:
        return "monitor", overall
    return "exclude", overall


def merge_scores(companies: list[ResolvedCompany], scores: list[CandidateScore]) -> list[ScoredCompany]:
    scores_by_url = {s.company_url: s for s in scores}
    merged: list[ScoredCompany] = []
    for company in companies:
        score = scores_by_url.get(company.primary_url)
        if score is None:
            # Already surfaced as a coverage gap by the caller — can't build
            # a ScoredCompany without real score data, so this company is
            # dropped from the scored output rather than faked with a default.
            continue
        rule_recommendation, rule_score = compute_recommendation(
            score.product_overlap, score.export_orientation, score.info_availability
        )
        merged.append(
            ScoredCompany(
                **company.model_dump(),
                reasoning=score.reasoning,
                product_overlap=score.product_overlap,
                export_orientation=score.export_orientation,
                info_availability=score.info_availability,
                llm_recommendation=score.llm_recommendation,
                rule_recommendation=rule_recommendation,
                rule_score=rule_score,
                agrees=score.llm_recommendation == rule_recommendation,
            )
        )
    return merged


def run_discovery(llm_client: LLMClient, request: DiscoveryRequest) -> list[ResolvedCompany] | list[ScoredCompany]:
    # Own span for search+classify — mirrors agent_loop's pattern: a
    # distinct job gets its own grouping, so the trace reads as nested
    # spans, not a flat pile of tool/generation calls directly under the route.
    with get_client().start_as_current_observation(name="gather_candidates", as_type="span") as gather_span:
        verdicts, missing = gather_and_classify_candidates(llm_client)
        gather_span.update(metadata={"candidate_count": len(verdicts)})

    if missing:
        # Surfaced on the currently-active span (the route's root), not
        # swallowed — see check_full_coverage.
        get_client().update_current_span(level="WARNING", metadata={"extraction_coverage_gap": sorted(missing)})

    companies = deduplicate_candidates(llm_client, verdicts)

    with get_client().start_as_current_observation(name="resolve_homepages", as_type="span"):
        resolved = resolve_primary_urls(companies)

    if not request.score_candidates:
        return resolved

    with get_client().start_as_current_observation(name="score_candidates", as_type="span") as score_span:
        profiles = gather_company_profiles(resolved)
        scores, missing_scores = score_companies(llm_client, resolved, profiles, request.our_products)
        score_span.update(metadata={"scored_count": len(scores)})

    if missing_scores:
        get_client().update_current_span(level="WARNING", metadata={"scoring_coverage_gap": sorted(missing_scores)})

    return merge_scores(resolved, scores)
