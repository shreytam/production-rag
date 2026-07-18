"""Langfuse-backed evaluation seam.

Defines the normalized types the eval runner/gate/CLI use and the `EvalBackend`
protocol that isolates the Langfuse SDK. `langfuse` is imported lazily inside the
real backend ONLY, so importing this module (and thus lint/offline tests) needs
neither the server nor the package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_output: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)
    tenant_id: str = "public"


@dataclass
class RunItemScores:
    item_id: str
    scores: dict[str, float]


# task: GoldenItem -> pipeline output dict (answer, retrieved_ids, contexts, ...)
Task = Callable[[GoldenItem], dict]
# evaluator: (item, output) -> [(score_name, value), ...]
Evaluator = Callable[[GoldenItem, dict], list[tuple[str, float]]]


@runtime_checkable
class EvalBackend(Protocol):
    def ensure_dataset(self, name: str) -> None: ...
    def upsert_item(self, *, dataset: str, item: GoldenItem) -> None: ...
    def get_dataset_items(self, name: str) -> list[GoldenItem]: ...
    def add_item_from_trace(self, *, dataset: str, trace_id: str) -> None: ...
    def run_experiment(
        self,
        *,
        dataset: str,
        run_name: str,
        items: list[GoldenItem],
        task: Task,
        evaluators: list[Evaluator],
        max_concurrency: int = 8,
    ) -> None: ...
    def get_run_scores(self, *, dataset: str, run_name: str) -> list[RunItemScores]: ...


def build_backend(settings=None) -> EvalBackend:
    """Construct the real Langfuse-backed backend. Imports langfuse lazily."""
    from eval._langfuse_backend import LangfuseEvalBackend

    return LangfuseEvalBackend(settings=settings)
