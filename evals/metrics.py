"""Pure scoring logic for the discovery eval — no API calls, so it's unit
tested in tests/test_eval_metrics.py like the rest of the pure functions."""

from statistics import mean
from urllib.parse import urlparse

from app.diagnostics import ResponseWarning
from app.discovery_models import ScoredCompany

VALID_EXPECTED = {"deep_dive", "monitor", "exclude", None}


def load_golden(data: dict) -> tuple[str, list[dict]]:
    our_products = data.get("our_products", "").strip()
    competitors = data.get("competitors", [])
    if not our_products:
        raise ValueError("golden set needs a non-empty 'our_products'")
    if not competitors:
        raise ValueError("golden set has no competitors yet — fill it in first (format: evals/golden_set.example.json)")
    for entry in competitors:
        if not entry.get("name"):
            raise ValueError(f"golden entry missing 'name': {entry}")
        if entry.get("expected") not in VALID_EXPECTED:
            raise ValueError(f"golden entry {entry['name']!r} has invalid 'expected': {entry.get('expected')!r}")
    return our_products, competitors


def normalize_domain(url_or_domain: str) -> str:
    # Accepts a bare domain ("acme.com") or a full URL; strips scheme, port
    # and a leading "www." so the golden set doesn't have to match exactly.
    netloc = urlparse(url_or_domain).netloc if "://" in url_or_domain else url_or_domain
    netloc = netloc.lower().split(":")[0]
    return netloc.removeprefix("www.")


def match_reason(entry: dict, company: ScoredCompany) -> str | None:
    # Domain first: a homepage domain is far less ambiguous than a name, and
    # it survives the Chinese/English naming split that names don't.
    golden_domains = {normalize_domain(d) for d in entry.get("domains", [])}
    for url in [company.primary_url, *company.urls]:
        found = normalize_domain(url)
        if any(found == d or found.endswith("." + d) for d in golden_domains):
            return "domain"
    name = company.canonical_name.casefold()
    for alias in [entry["name"], *entry.get("aliases", [])]:
        if alias.strip() and alias.casefold() in name:
            return "name"
    return None


def score_run(
    competitors: list[dict], companies: list[ScoredCompany], warnings: list[ResponseWarning] = ()
) -> dict:
    matched, missed, duplicates = [], [], []
    matched_company_ids: set[int] = set()
    for entry in competitors:
        hits = [(c, reason) for c in companies if (reason := match_reason(entry, c))]
        if not hits:
            missed.append(entry["name"])
            continue
        if len(hits) > 1:
            # One real company surfacing as several results is a dedup miss —
            # tracked on its own so a dedup fix shows up as a number.
            duplicates.append({"golden": entry["name"], "found_as": [c.canonical_name for c, _ in hits]})
        matched_company_ids.update(id(c) for c, _ in hits)
        best, reason = max(hits, key=lambda hit: hit[0].rule_score)
        matched.append(
            {
                "golden": entry["name"],
                "found_as": best.canonical_name,
                "matched_by": reason,
                "expected": entry.get("expected"),
                "rule_recommendation": best.rule_recommendation,
                "llm_recommendation": best.llm_recommendation,
                "rule_score": round(best.rule_score, 2),
            }
        )

    labeled = [m for m in matched if m["expected"] is not None]
    return {
        "found_count": len(companies),
        "golden_total": len(competitors),
        "recall": len(matched) / len(competitors),
        # Share of the whole golden set that both got found AND landed in
        # deep_dive — the number that matters if you only read the top tier.
        "deep_dive_recall": sum(m["rule_recommendation"] == "deep_dive" for m in matched) / len(competitors),
        "label_accuracy": (
            sum(m["rule_recommendation"] == m["expected"] for m in labeled) / len(labeled) if labeled else None
        ),
        "agreement_rate": sum(c.agrees for c in companies) / len(companies) if companies else None,
        "deep_dive_count": sum(c.rule_recommendation == "deep_dive" for c in companies),
        "duplicate_count": len(duplicates),
        # The pipeline's own report of what made this run incomplete (search
        # failures, unscored companies...). Recall on a run with warnings
        # partly measures the outage, not the code.
        "warnings": [w.model_dump() for w in warnings],
        "matched": matched,
        "missed": missed,
        "duplicates": duplicates,
        # Not in the golden set: either noise, or a real competitor worth
        # adding to it — review these when growing the golden set.
        "unmatched_found": [c.canonical_name for c in companies if id(c) not in matched_company_ids],
    }


def is_valid_run(run: dict) -> bool:
    # search_unavailable isn't a bad result, it's no result (e.g. an
    # exhausted Tavily quota). Averaging it in as recall 0 would poison the
    # summary. found_count > 0 stays as a backstop for runs saved before the
    # pipeline reported warnings.
    codes = {w["code"] for w in run.get("warnings", [])}
    return run["found_count"] > 0 and "search_unavailable" not in codes


def aggregate_runs(all_runs: list[dict]) -> dict:
    runs = [r for r in all_runs if is_valid_run(r)]

    def spread(key: str) -> dict:
        values = [r[key] for r in runs if r[key] is not None]
        if not values:
            return {"mean": None, "min": None, "max": None}
        return {"mean": round(mean(values), 3), "min": round(min(values), 3), "max": round(max(values), 3)}

    # Per golden company: how much its rule_score moves between identical
    # runs. Large swings mean the ranking isn't trustworthy run to run.
    scores_by_company: dict[str, list[float]] = {}
    for run in runs:
        for m in run["matched"]:
            scores_by_company.setdefault(m["golden"], []).append(m["rule_score"])
    score_stability = {
        name: {"found_in_runs": len(scores), "min": min(scores), "max": max(scores)}
        for name, scores in scores_by_company.items()
    }

    return {
        "runs": len(runs),
        "invalid_runs": len(all_runs) - len(runs),
        "recall": spread("recall"),
        "deep_dive_recall": spread("deep_dive_recall"),
        "label_accuracy": spread("label_accuracy"),
        "agreement_rate": spread("agreement_rate"),
        "found_count": spread("found_count"),
        "duplicate_matches_per_run": spread("duplicate_count"),
        "score_stability": score_stability,
    }
