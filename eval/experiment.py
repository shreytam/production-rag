"""Langfuse-native experiment runner.

Builds the task (pipeline call) + evaluators and submits them via the backend's
run_experiment. Dependency-injected `run(...)` is unit-testable with a fake
backend/pipeline; `main()` wires the real backend, pipeline, and models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from eval.evaluators import build_evaluators
from eval.fast_subset import fast_subset
from eval.langfuse_eval import EvalBackend, GoldenItem, Task


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_task(pipeline) -> Task:
    from core.types import ACLContext

    def task(item: GoldenItem) -> dict:
        acl = ACLContext(tenant_id=item.tenant_id)
        return pipeline.run(item.question, acl=acl)

    return task


def run(*, backend: EvalBackend, pipeline, generator, embedder, dataset: str,
        run_name: str, limit: int | None = None, fast: bool = False,
        judge_votes: int = 1, judge_seed: int = 0, max_concurrency: int = 8) -> str:
    from core.config import get_settings

    settings = get_settings()
    items = backend.get_dataset_items(dataset)
    if not items:
        print(f"[experiment] Dataset empty or not found: {dataset}", file=sys.stderr)
        sys.exit(1)
    if limit:
        items = items[:limit]
    if fast:
        items = fast_subset(items, n=settings.eval_fast_n, seed=settings.eval_fast_seed)

    task = build_task(pipeline)
    evaluators = build_evaluators(generator, embedder,
                                  judge_votes=judge_votes, judge_seed=judge_seed)
    backend.run_experiment(dataset=dataset, run_name=run_name, items=items,
                           task=task, evaluators=evaluators,
                           max_concurrency=max_concurrency)
    print(f"[experiment] Ran {len(items)} items as run '{run_name}' on dataset '{dataset}'")
    return run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Langfuse eval experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", choices=["baseline", "full"], default="full")
    parser.add_argument("--run-name", default=None,
                        help="Defaults to '<version>@<git_sha>'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()

    from core.config import get_settings
    from core.pipeline import build as build_pipeline
    from core.registry import build_embedder, build_generator
    from eval.langfuse_eval import build_backend

    settings = get_settings()
    run_name = args.run_name or f"{args.version}@{_git_sha()}-{int(time.time())}"
    pipeline = build_pipeline(version=args.version, corpus=args.dataset,
                              enable_guardrails=False, enable_cache=False)
    run(
        backend=build_backend(settings),
        pipeline=pipeline,
        generator=build_generator(role="judge"),
        embedder=build_embedder(),
        dataset=args.dataset,
        run_name=run_name,
        limit=args.limit,
        fast=args.fast,
        judge_votes=settings.judge_votes,
        judge_seed=settings.judge_seed,
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    main()
