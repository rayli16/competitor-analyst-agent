"""Discovery eval: run the real /discover pipeline against a hand-labeled
golden set and report recall, ranking quality, and run-to-run stability.

Calls the real Anthropic and Tavily APIs — costs money (~$0.25 per run per
the README's Langfuse numbers), so it's run by hand, never in CI.

    python -m evals.run_discovery_eval            # one run
    python -m evals.run_discovery_eval --runs 3   # stability across runs

The real golden set (evals/golden_set.json) is internal business data and
gitignored — create it locally in the format of evals/golden_set.example.json.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langfuse import get_client

from app.discovery import run_discovery
from app.discovery_models import DiscoveryRequest
from app.llm import LLMClient
from evals.metrics import aggregate_runs, is_valid_run, load_golden, score_run

EVALS_DIR = Path(__file__).parent


def print_run(index: int, run: dict, seconds: float) -> None:
    print(f"\n=== run {index}  ({seconds:.0f}s) ===")
    print(f"found {run['found_count']} companies, {run['deep_dive_count']} rule deep_dive")
    print(f"recall            {run['recall']:.0%}  ({len(run['matched'])}/{run['golden_total']})")
    print(f"deep_dive recall  {run['deep_dive_recall']:.0%}")
    if run["label_accuracy"] is not None:
        print(f"label accuracy    {run['label_accuracy']:.0%}")
    if run["agreement_rate"] is not None:
        print(f"llm/rule agree    {run['agreement_rate']:.0%}")
    if run["missed"]:
        print("missed:           " + ", ".join(run["missed"]))
    for dup in run["duplicates"]:
        print(f"duplicate:        {dup['golden']} -> {dup['found_as']}")
    # The pipeline's own warnings — any of them means these numbers partly
    # reflect an incomplete run, not just the code under test.
    for warning in run["warnings"]:
        print(f"WARNING [{warning['code']}]: {warning['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="identical runs to execute (>1 measures stability)")
    parser.add_argument("--golden", type=Path, default=EVALS_DIR / "golden_set.json")
    args = parser.parse_args()

    if not args.golden.exists():
        parser.error(
            f"{args.golden} not found — it's gitignored internal data, so create it locally "
            f"using {EVALS_DIR / 'golden_set.example.json'} as the format"
        )
    # Fails before spending anything if the golden set isn't filled in.
    our_products, competitors = load_golden(json.loads(args.golden.read_text(encoding="utf-8")))

    llm_client = LLMClient()
    request = DiscoveryRequest(our_products=our_products, score_candidates=True)
    runs, raw = [], []
    for i in range(1, args.runs + 1):
        started = time.monotonic()
        # Own root span per run, so eval traffic is easy to find (and filter
        # out) in Langfuse next to real requests.
        with get_client().start_as_current_observation(
            name="eval_discovery", as_type="span", metadata={"run": i, "of": args.runs}
        ):
            companies, warnings = run_discovery(llm_client, request)
        seconds = time.monotonic() - started
        run = score_run(competitors, companies, warnings)
        run["seconds"] = round(seconds, 1)
        runs.append(run)
        raw.append([c.model_dump() for c in companies])
        if not is_valid_run(run):
            reasons = "; ".join(w["message"] for w in run["warnings"]) or "0 companies found"
            print(
                f"\n=== run {i}  INVALID ===\n{reasons}\n"
                "Excluded from the summary; stopping so the remaining runs don't fail the same way."
            )
            break
        print_run(i, run, seconds)

    summary = aggregate_runs(runs)
    if len(runs) > 1:
        print("\n=== across runs ===")
        print(json.dumps({k: v for k, v in summary.items() if k != "score_stability"}, indent=2))

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"discovery_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {"golden_file": str(args.golden), "our_products": our_products, "summary": summary, "runs": runs, "raw": raw},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {out_path}")

    # A short-lived script exits before Langfuse's background exporter
    # sends — flush explicitly or the eval traces never show up.
    get_client().flush()


if __name__ == "__main__":
    main()
