"""Langfuse-native regression gate.

Fetches per-item scores for the new run and a baseline run, aligns them by
dataset-item id, and applies paired-bootstrap and/or absolute-threshold checks
per the configured mode. Exits nonzero on failure.
"""

from __future__ import annotations

import argparse
import math
import sys

from eval.langfuse_eval import EvalBackend, RunItemScores
from eval.stats import paired_bootstrap


def _aligned_values(base: list[RunItemScores], new: list[RunItemScores]) -> dict[str, tuple[list[float], list[float]]]:
    """Return {metric: (base_values, new_values)} aligned by item id (sorted).

    Raises ValueError if the item sets differ or a metric is missing on one side.
    """
    base_by_id = {r.item_id: r.scores for r in base}
    new_by_id = {r.item_id: r.scores for r in new}
    if set(base_by_id) != set(new_by_id):
        raise ValueError(
            f"item-set mismatch: baseline has {sorted(base_by_id)}, new has {sorted(new_by_id)}"
        )
    ids = sorted(base_by_id)
    metrics = set().union(*(set(s) for s in base_by_id.values())) if ids else set()
    out: dict[str, tuple[list[float], list[float]]] = {}
    for metric in sorted(metrics):
        b_vals, n_vals = [], []
        for i in ids:
            if metric not in base_by_id[i] or metric not in new_by_id[i]:
                raise ValueError(f"metric '{metric}' missing on item '{i}'")
            b_vals.append(float(base_by_id[i][metric]))
            n_vals.append(float(new_by_id[i][metric]))
        out[metric] = (b_vals, n_vals)
    return out


def evaluate_gate(*, backend: EvalBackend, dataset: str, new_run: str,
                  baseline_run: str, mode: str = "bootstrap", tolerance: float = 0.03,
                  thresholds: dict[str, float] | None = None,
                  resamples: int = 1000) -> bool:
    thresholds = thresholds or {}
    base = backend.get_run_scores(dataset=dataset, run_name=baseline_run)
    new = backend.get_run_scores(dataset=dataset, run_name=new_run)
    aligned = _aligned_values(base, new)

    header = f"{'Metric':<22}{'Base':>10}{'New':>10}{'Delta':>10}{'CI hi':>10}{'Floor':>10}{'Verdict':>10}"
    sep = "-" * len(header)
    print(sep); print(header); print(sep)

    failures: list[str] = []
    for metric, (b_vals, n_vals) in aligned.items():
        base_mean = sum(b_vals) / len(b_vals)
        new_mean = sum(n_vals) / len(n_vals)
        delta = new_mean - base_mean
        verdict = "PASSED"
        ci_hi = float("nan")

        if mode in ("bootstrap", "both"):
            _, _lo, ci_hi = paired_bootstrap(b_vals, n_vals, n=resamples)
            if math.isnan(ci_hi) or ci_hi < -tolerance:
                verdict = "FAILED"
                failures.append(f"{metric}: regression CI hi {ci_hi:.4f} < {-tolerance:.4f}")

        floor = thresholds.get(metric, float("nan"))
        if mode in ("threshold", "both") and metric in thresholds:
            if new_mean < floor:
                verdict = "FAILED"
                failures.append(f"{metric}: mean {new_mean:.4f} < floor {floor:.4f}")

        print(f"{metric:<22}{base_mean:>10.4f}{new_mean:>10.4f}{delta:>10.4f}"
              f"{ci_hi:>10.4f}{floor:>10.4f}{verdict:>10}")

    print(sep)
    if failures:
        print("\n[gate] GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return False
    print("\n[gate] Gate PASSED.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse eval regression gate.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--baseline-run", default=None,
                        help="Defaults to Settings.eval_baseline_run.")
    parser.add_argument("--mode", choices=["bootstrap", "threshold", "both"], default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()

    from core.config import get_settings
    from eval.langfuse_eval import build_backend

    settings = get_settings()
    passed = evaluate_gate(
        backend=build_backend(settings),
        dataset=args.dataset,
        new_run=args.new_run,
        baseline_run=args.baseline_run or settings.eval_baseline_run,
        mode=args.mode or settings.eval_gate_mode,
        tolerance=args.tolerance if args.tolerance is not None else settings.eval_tolerance,
        thresholds=settings.eval_gate_thresholds,
        resamples=settings.eval_bootstrap_resamples,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
