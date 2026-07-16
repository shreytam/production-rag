"""CLI evaluation runner.

Usage::

    python -m eval.run_eval --dataset squad_dev --version baseline [--fast] [--limit 50]

Outputs ``eval/runs/{dataset}.{version}.results.json`` containing per-item
metrics, aggregates, bootstrap confidence intervals, and provenance metadata.

The ``core.pipeline`` module is imported lazily so this module remains
importable even before the pipeline workstream is complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.fast_subset import fast_subset
from eval.generation_metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from eval.llm_judge import holistic_judge
from eval.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from eval.stats import bootstrap_ci

RUNS_DIR = Path(__file__).parent / "runs"
BASELINES_DIR = Path(__file__).parent / "baselines"


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _model_versions(generator: Any, embedder: Any) -> dict:
    versions: dict[str, str] = {}
    for attr in ("model", "model_name", "model_id"):
        if hasattr(generator, attr):
            versions["generator_model"] = str(getattr(generator, attr))
            break
    for attr in ("model", "model_name", "model_id"):
        if hasattr(embedder, attr):
            versions["embedder_model"] = str(getattr(embedder, attr))
            break
    return versions


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def _build_pipeline(version: str, corpus: str | None = None) -> Any:
    """Build the pipeline for the given version, tolerating missing module."""
    try:
        import core.pipeline as pipeline_mod  # noqa: F401 – built by another workstream
        # Guardrails OFF for eval: CitationGuardrail/SchemaGuardrail would BLOCK
        # normal answers (confounding metrics) and groundedness adds an LLM call.
        return pipeline_mod.build(version=version, corpus=corpus, enable_guardrails=False)
    except ImportError as exc:
        print(
            f"[run_eval] core.pipeline is not yet available ({exc}). "
            "Cannot run end-to-end evaluation. "
            "Ensure the pipeline workstream has been merged.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------


def evaluate_item(
    item: dict,
    pipeline: Any,
    generator: Any,
    embedder: Any,
    compute_gen: bool = True,
) -> dict:
    """Run the pipeline + metrics for a single golden item.

    Expected golden item keys:
        question (str), ground_truth (str), relevant_chunk_ids (list[str])

    When ``compute_gen`` is False, only the cheap retrieval metrics (pure set
    math, ~1 pipeline call) are computed — used for a tight retrieval-only pass
    over the full golden set without the ~5-LLM-calls-per-item RAGAS/judge cost.
    """
    from core.types import ACLContext

    question: str = item["question"]
    ground_truth: str = item.get("ground_truth", "")
    relevant_ids: set[str] = set(item.get("relevant_chunk_ids", []))

    # Run pipeline within the item's tenant scope (default "public").
    acl = ACLContext(tenant_id=item.get("tenant_id", "public"))
    result = pipeline.run(question, acl=acl)
    answer_text: str = result.get("answer", "")
    retrieved_ids: list[str] = result.get("retrieved_ids", [])
    context_texts: list[str] = result.get("contexts", [])

    # Retrieval metrics
    ret_metrics = {
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
    }

    out: dict = {
        "question": question,
        "answer": answer_text,
        "retrieved_ids": retrieved_ids,
        "retrieval_metrics": ret_metrics,
        "generation_metrics": {},
        "judge": None,
    }
    if not compute_gen:
        return out

    from core.config import get_settings
    settings = get_settings()

    out["generation_metrics"] = {
        "faithfulness": faithfulness(question, answer_text, context_texts, generator),
        "answer_relevancy": answer_relevancy(question, answer_text, generator, embedder),
        "context_precision": context_precision(question, answer_text, context_texts, generator),
        "context_recall": context_recall(question, ground_truth, context_texts, generator),
    }
    out["judge"] = holistic_judge(
        question,
        answer_text,
        context_texts,
        generator,
        votes=settings.judge_votes,
        base_seed=settings.judge_seed,
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(items: list[dict], bootstrap_resamples: int = 1000) -> dict:
    """Compute mean + bootstrap CI for every numeric metric."""
    has_judge = items[0].get("judge") is not None
    metric_keys = (
        list(items[0]["retrieval_metrics"].keys())
        + list(items[0].get("generation_metrics", {}).keys())
        + (["judge_score"] if has_judge else [])
    )

    def _collect(key: str) -> list[float]:
        values = []
        for it in items:
            if key in it["retrieval_metrics"]:
                values.append(it["retrieval_metrics"][key])
            elif key in it["generation_metrics"]:
                values.append(it["generation_metrics"][key])
            elif key == "judge_score":
                values.append(it["judge"]["score"])
        return values

    aggregates: dict[str, dict] = {}
    for key in metric_keys:
        vals = _collect(key)
        mean, lo, hi = bootstrap_ci(vals, n=bootstrap_resamples)
        aggregates[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    return aggregates


# ---------------------------------------------------------------------------
# Main entry point (importable for testing)
# ---------------------------------------------------------------------------


def run_eval(
    dataset: str,
    version: str,
    fast: bool = False,
    limit: int | None = None,
    skip_gen_metrics: bool = False,
    write_baseline: bool = False,
) -> Path:
    """Run evaluation and write results JSON.  Returns the output path."""
    from core.registry import build_embedder, build_generator

    if write_baseline and not fast:
        print("[run_eval] Error: Requires --fast option when --write-baseline is active.", file=sys.stderr)
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "eval" / f"{dataset}.json"
    if not dataset_path.exists():
        print(f"[run_eval] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with dataset_path.open() as f:
        items: list[dict] = json.load(f)

    from core.config import get_settings
    settings = get_settings()

    if limit:
        items = items[:limit]
    if fast:
        items = fast_subset(items, n=settings.eval_fast_n, seed=settings.eval_fast_seed)

    pipeline = _build_pipeline(version, corpus=dataset)
    generator = build_generator(role="judge")
    embedder = build_embedder()

    evaluated = []
    n = len(items)
    for i, item in enumerate(items, start=1):
        t_item = time.time()
        evaluated.append(
            evaluate_item(item, pipeline, generator, embedder, compute_gen=not skip_gen_metrics)
        )
        print(
            f"[run_eval] item {i}/{n} done in {time.time() - t_item:.1f}s "
            f"(recall@5={evaluated[-1]['retrieval_metrics']['recall_at_5']:.2f})",
            flush=True,
        )
    aggregates = aggregate_results(evaluated, bootstrap_resamples=settings.eval_bootstrap_resamples)

    output = {
        "dataset": dataset,
        "version": version,
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_versions": _model_versions(generator, embedder),
        "n_items": len(evaluated),
        "aggregates": aggregates,
        "items": evaluated,
    }

    # Resolve output path
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    if write_baseline:
        out_path = BASELINES_DIR / f"{dataset}.json"
    else:
        suffix = ".retrieval" if skip_gen_metrics else ""
        out_path = RUNS_DIR / f"{dataset}.{version}{suffix}.results.json"

    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[run_eval] Wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline.")
    parser.add_argument("--dataset", required=True, help="Dataset name (matches data/eval/<name>.json)")
    parser.add_argument("--version", choices=["baseline", "full"], default="full")
    parser.add_argument("--fast", action="store_true", help="Use fast_subset (15 items)")
    parser.add_argument("--limit", type=int, default=None, help="Hard cap on number of items")
    parser.add_argument(
        "--skip-gen-metrics",
        action="store_true",
        help="Retrieval metrics only (skip RAGAS + judge LLM calls) — fast full-set pass",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write results directly to baseline JSON path.",
    )
    args = parser.parse_args()

    run_eval(
        dataset=args.dataset,
        version=args.version,
        fast=args.fast,
        limit=args.limit,
        skip_gen_metrics=args.skip_gen_metrics,
        write_baseline=args.write_baseline,
    )


if __name__ == "__main__":
    main()
