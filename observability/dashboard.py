"""Offline dashboard — aggregates eval run results and prints a compact summary.

The live UI is provided by Langfuse.  This module handles the *offline* case:
reading ``results.json`` artifacts produced by ``eval/`` runs and computing
aggregate statistics for quick inspection from the CLI or CI logs.

Public surface
--------------
summarize_runs(results_json_paths) -> dict
    Read one or more results.json files and return aggregate stats.

print_dashboard(summary) -> None
    Print a compact text panel to stdout.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def summarize_runs(results_json_paths: list[str | Path]) -> dict[str, Any]:
    """Aggregate latency, cost, and metric summaries across eval run files.

    Each ``results.json`` is expected to be a list of run-record dicts, where
    each record may contain the keys:

    - ``latency_ms``     (float) — end-to-end query latency
    - ``cost_usd``       (float) — per-query cost estimate
    - ``metrics``        (dict[str, float]) — named quality metrics (e.g. faithfulness)

    Missing keys are silently skipped.

    Returns
    -------
    dict with keys:
        n_runs      — total number of run records read
        latency_ms  — {mean, median, p95, min, max} or {} if no data
        cost_usd    — {mean, total} or {} if no data
        metrics     — {metric_name: {mean, min, max}} for each metric seen
    """
    latencies: list[float] = []
    costs: list[float] = []
    metric_values: dict[str, list[float]] = {}
    n_runs = 0

    for path in results_json_paths:
        raw = Path(path).read_text(encoding="utf-8")
        records = json.loads(raw)
        if isinstance(records, dict):
            # Allow a single record or a wrapper with a "results" key
            records = records.get("results", [records])

        for rec in records:
            n_runs += 1
            if "latency_ms" in rec:
                latencies.append(float(rec["latency_ms"]))
            if "cost_usd" in rec:
                costs.append(float(rec["cost_usd"]))
            for k, v in rec.get("metrics", {}).items():
                metric_values.setdefault(k, []).append(float(v))

    def _stats_numeric(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        sorted_v = sorted(values)
        p95_idx = max(0, int(len(sorted_v) * 0.95) - 1)
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_v[p95_idx],
            "min": min(values),
            "max": max(values),
        }

    def _metric_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    summary: dict[str, Any] = {
        "n_runs": n_runs,
        "latency_ms": _stats_numeric(latencies),
        "cost_usd": (
            {"mean": statistics.mean(costs), "total": sum(costs)} if costs else {}
        ),
        "metrics": {k: _metric_stats(v) for k, v in metric_values.items()},
    }
    return summary


def print_dashboard(summary: dict[str, Any]) -> None:
    """Print a compact text panel summarising eval run results."""
    width = 60
    sep = "─" * width

    print(sep)
    print(f"  RAG Eval Dashboard  ({summary['n_runs']} runs)")
    print(sep)

    lat = summary.get("latency_ms", {})
    if lat:
        print(f"  Latency (ms)  mean={lat['mean']:.1f}  median={lat['median']:.1f}"
              f"  p95={lat['p95']:.1f}  min={lat['min']:.1f}  max={lat['max']:.1f}")
    else:
        print("  Latency       (no data)")

    cost = summary.get("cost_usd", {})
    if cost:
        print(f"  Cost (USD)    mean={cost['mean']:.6f}  total={cost['total']:.6f}")
    else:
        print("  Cost          (no data)")

    metrics = summary.get("metrics", {})
    if metrics:
        print("  Metrics:")
        for name, stats in metrics.items():
            print(f"    {name:<24} mean={stats['mean']:.4f}"
                  f"  min={stats['min']:.4f}  max={stats['max']:.4f}")
    else:
        print("  Metrics       (no data)")

    print(sep)
