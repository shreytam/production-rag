"""CLI tool for comparing two evaluation runs.

Usage::

    python -m eval.compare --dataset squad_dev --base baseline --new full
    python -m eval.compare --dataset squad_dev --base baseline --new full --tolerance 0.02
    python -m eval.compare --dataset squad_dev --baseline-file eval/baselines/squad_dev.json --new full

Prints a fixed-width metrics table (base / new / delta / CI) and exits nonzero
if any metric in *new* drops below *base - tolerance*.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import math
from eval.stats import paired_bootstrap

RUNS_DIR = Path(__file__).parent / "runs"
BASELINES_DIR = Path(__file__).parent / "baselines"


def _load_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


CUSTOM_EVAL_HOOKS = {}


def register_custom_metric(name, scoring_func):
    """Registry seam to hook custom evaluations into the gate pipeline."""
    CUSTOM_EVAL_HOOKS[name] = scoring_func


def _extract_aggregates(results: dict) -> dict[str, float]:
    """Return {metric_name: mean} from a results JSON, including registered custom evals."""
    aggs = {}
    for k, v in results.get("aggregates", {}).items():
        mean_val = v.get("mean") if isinstance(v, dict) else v
        if mean_val is not None and not (isinstance(mean_val, float) and math.isnan(mean_val)):
            aggs[k] = float(mean_val)
    return aggs


def _extract_item_values(results: dict, metric: str) -> list[float]:
    """Reconstruct per-item values for a given metric from the results JSON."""
    values = []
    for item in results.get("items", []):
        if metric in item.get("retrieval_metrics", {}):
            values.append(item["retrieval_metrics"][metric])
        elif metric in item.get("generation_metrics", {}):
            values.append(item["generation_metrics"][metric])
        elif metric == "judge_score":
            values.append(item["judge"]["score"])
    return values


def compare(
    dataset: str,
    base_version: str | None = None,
    new_version: str = "full",
    tolerance: float = 0.02,
    baseline_file: Path | None = None,
) -> bool:
    """Compare two runs and return True if all metrics pass the gate."""
    if baseline_file is not None:
        base_path = baseline_file
    elif base_version is not None:
        base_path = RUNS_DIR / f"{dataset}.{base_version}.results.json"
    else:
        base_path = BASELINES_DIR / f"{dataset}.json"

    new_path = RUNS_DIR / f"{dataset}.{new_version}.results.json"

    for p in (base_path, new_path):
        if not p.exists():
            print(f"[compare] Results file not found: {p}", file=sys.stderr)
            sys.exit(1)

    base_results = _load_results(base_path)
    new_results = _load_results(new_path)

    base_agg = _extract_aggregates(base_results)
    new_agg = _extract_aggregates(new_results)

    all_metrics = sorted(set(base_agg.keys()) | set(new_agg.keys()) | set(CUSTOM_EVAL_HOOKS.keys()))
    rows = []
    failures: list[str] = []

    # Print table header with Verdict
    header = f"{'Metric':<25}{'Base':>10}{'New':>10}{'Delta':>10}{'CI lo':>10}{'CI hi':>10}{'Verdict':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for metric in all_metrics:
        # Check custom hook integration
        if metric in CUSTOM_EVAL_HOOKS:
            # Custom checks calculate scores from raw items
            hook = CUSTOM_EVAL_HOOKS[metric]
            base_items = [hook(item) for item in base_results.get("items", [])]
            new_items = [hook(item) for item in new_results.get("items", [])]
            base_val = sum(base_items) / len(base_items) if base_items else float("nan")
            new_val = sum(new_items) / len(new_items) if new_items else float("nan")
        else:
            base_val = base_agg.get(metric, float("nan"))
            new_val = new_agg.get(metric, float("nan"))

            base_items = _extract_item_values(base_results, metric)
            new_items = _extract_item_values(new_results, metric)

        # Enforce equal-N check
        if not base_items or not new_items:
            print(f"[compare] Error: Missing values for metric {metric}", file=sys.stderr)
            sys.exit(1)
        if len(base_items) != len(new_items):
            print(f"[compare] Error: Sample size N mismatch for {metric}. Base has {len(base_items)}, New has {len(new_items)}", file=sys.stderr)
            sys.exit(1)

        # Gating checks
        delta = new_val - base_val
        if math.isnan(base_val) or math.isnan(new_val):
            verdict = "FAILED"
            failures.append(f"{metric}: Base or New is NaN")
            lo = hi = float("nan")
        else:
            # Run paired bootstrap
            _, lo, hi = paired_bootstrap(base_items, new_items)

            # Gating Rule: fail if statistically significant regression
            if math.isnan(lo) or math.isnan(hi):
                verdict = "FAILED"
                failures.append(f"{metric}: bootstrap returned NaN confidence intervals")
            elif hi < -tolerance:
                verdict = "FAILED"
                failures.append(f"{metric}: regression confidence bound {hi:.4f} < {-tolerance:.4f}")
            else:
                verdict = "PASSED"

        print(
            f"{metric:<25}{base_val:>10.4f}{new_val:>10.4f}{delta:>10.4f}{lo:>10.4f}{hi:>10.4f}{verdict:>12}"
        )
        rows.append((metric, base_val, new_val, delta, lo, hi, verdict))

    print(sep)

    if failures:
        print("\n[compare] GATE FAILED — metric regressions detected:")
        for f in failures:
            print(f"  - {f}")
        return False

    print("\n[compare] All metrics within tolerance. Gate PASSED.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RAG evaluation runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", default=None, help="Base version name (e.g. 'baseline')")
    parser.add_argument("--new", default="full", help="New version name (e.g. 'full')")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Explicit path to baseline results JSON (overrides --base).",
    )
    args = parser.parse_args()

    passed = compare(
        dataset=args.dataset,
        base_version=args.base,
        new_version=args.new,
        tolerance=args.tolerance,
        baseline_file=args.baseline_file,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
