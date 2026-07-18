"""In-memory EvalBackend for offline tests."""

from __future__ import annotations

from eval.langfuse_eval import Evaluator, GoldenItem, RunItemScores, Task


class FakeEvalBackend:
    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, GoldenItem]] = {}
        self.runs: dict[tuple[str, str], list[RunItemScores]] = {}
        self.trace_items: dict[str, list[str]] = {}

    def ensure_dataset(self, name: str) -> None:
        self.datasets.setdefault(name, {})

    def upsert_item(self, *, dataset: str, item: GoldenItem) -> None:
        self.datasets.setdefault(dataset, {})[item.id] = item

    def get_dataset_items(self, name: str) -> list[GoldenItem]:
        return list(self.datasets.get(name, {}).values())

    def add_item_from_trace(self, *, dataset: str, trace_id: str) -> None:
        self.trace_items.setdefault(dataset, []).append(trace_id)

    def run_experiment(self, *, dataset: str, run_name: str, items: list[GoldenItem],
                       task: Task, evaluators: list[Evaluator],
                       max_concurrency: int = 8) -> None:
        results: list[RunItemScores] = []
        for item in items:
            output = task(item)
            scores: dict[str, float] = {}
            for ev in evaluators:
                for name, value in ev(item, output):
                    scores[name] = float(value)
            results.append(RunItemScores(item_id=item.id, scores=scores))
        self.runs[(dataset, run_name)] = results

    def get_run_scores(self, *, dataset: str, run_name: str) -> list[RunItemScores]:
        if (dataset, run_name) not in self.runs:
            raise KeyError(f"run not found: {dataset}/{run_name}")
        return self.runs[(dataset, run_name)]
