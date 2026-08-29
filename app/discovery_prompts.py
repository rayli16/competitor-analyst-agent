from app.discovery_models import CandidateVerdict, ResolvedCompany

EXTRACTION_SYSTEM_PROMPT = (
    "You are classifying web search results for a competitor-discovery pipeline "
    "in the ophthalmic equipment industry. For every result provided, produce "
    "exactly one verdict — do not skip any, even ones you're confident are noise; "
    "classify them as such instead of omitting them. Base each classification on "
    "what's actually in the source, not assumptions. Watch specifically for "
    "third-party pages — news articles, investment writeups, directory or "
    "aggregator listings — that discuss a company without being that company's "
    "own site; these are not the company, no matter how much detail they give "
    "about one."
)


def build_extraction_message(results: list[dict[str, str]]) -> str:
    sections = [f"[{i}] {r['title']} ({r['url']})\n{r['content']}" for i, r in enumerate(results, start=1)]
    return f"Classify each of the following {len(results)} search results:\n\n" + "\n\n".join(sections)


DEDUP_SYSTEM_PROMPT = (
    "You are merging duplicate company entries in a competitor-discovery pipeline. "
    "The same real company sometimes appears under multiple URLs — a corporate site, "
    "a product-line-specific site, a regional site — sometimes under completely "
    "different names (a legal entity name on one, a brand name on another). Merge "
    "entries only when you have real evidence they're the same company: a shared "
    "address or phone number appearing in both sources, matching legal/registration "
    "names, near-identical 'About Us' text reused across sites, or an explicit "
    "cross-reference ('our other site is...', 'our brand X'). When it's ambiguous — "
    "you can't point to a specific shared signal — do NOT merge; keep them as "
    "separate entries. An incorrect merge silently drops a real competitor from this "
    "pipeline; a missed merge only costs one redundant follow-up review, so err "
    "toward keeping entries separate when unsure."
)


def build_dedup_message(candidates: list[CandidateVerdict]) -> str:
    sections = [f"- {v.company_name} ({v.url}) [{v.entity_type}]\n  {v.evidence}" for v in candidates]
    return f"Merge duplicate companies among these {len(candidates)} candidates:\n\n" + "\n".join(sections)


SCORING_SYSTEM_PROMPT = (
    "You are scoring competitor candidates against our own product lines for a "
    "competitor-intelligence funnel. For every company provided, produce exactly "
    "one score — do not skip any. Score product_overlap by product category and "
    "function, not exact naming — a company selling equivalent equipment under "
    "different Chinese/English terminology still counts as overlap. Score "
    "export_orientation on actual evidence of exporting or serving overseas "
    "markets, not company size or reputation alone. Score info_availability on "
    "how much the profile evidence below actually contains — a company with a "
    "real homepage and specific details scores high on this dimension regardless "
    "of how relevant it otherwise is."
)


def build_scoring_message(
    companies: list[ResolvedCompany], profiles: dict[str, list[dict[str, str]]], our_products: str, content_truncate_chars: int
) -> str:
    sections = []
    for company in companies:
        results = profiles.get(company.primary_url, [])
        evidence = (
            "\n\n".join(f"{r['title']} ({r['url']})\n{r['content'][:content_truncate_chars]}" for r in results)
            or "(no profile search results found)"
        )
        sections.append(
            f"Company: {company.canonical_name} ({company.primary_url})\n"
            f"Entity type: {company.entity_type}\n"
            f"Evidence:\n{evidence}"
        )
    return (
        f"Our products: {our_products}\n\n"
        f"Score each of the following {len(companies)} companies:\n\n" + "\n\n---\n\n".join(sections)
    )
