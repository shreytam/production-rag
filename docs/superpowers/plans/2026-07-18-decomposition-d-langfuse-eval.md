# Decomposition D — Langfuse-native Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local file-based eval runner + gate with a Langfuse-native
workflow: datasets and experiment runs live on a hosted Langfuse, driven by the
SDK's `run_experiment`, with a configurable regression gate and a trace-curation CLI.

**Architecture:** A single `EvalBackend` protocol isolates the Langfuse SDK (the
only module importing `langfuse`). `experiment.py` builds a `task` (pipeline call)
and `evaluators` (thin wrappers over existing metric functions) and submits them via
`backend.run_experiment`. `gate.py` reads per-item scores back from Langfuse and
applies paired-bootstrap and/or absolute-threshold checks. `dataset_cli.py` seeds
datasets and promotes traces to items. All logic is unit-tested against an in-memory
`FakeEvalBackend`; the real backend's live calls are exercised only by the CI eval job.

**Tech Stack:** Python 3.12, Langfuse SDK 4.9.1 (`run_experiment`, `get_dataset`,
`get_dataset_run`, `create_dataset_item`, `api.scores.get_many`), pydantic-settings,
numpy (existing `eval/stats.py`), pytest.

## Global Constraints

- Commit authorship: EVERY commit uses `git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "..."`. No Claude/AI attribution of any kind (no `Co-Authored-By`, no `Claude-Session`, no "Generated with" note).
- `langfuse` is imported LAZILY and ONLY inside `eval/langfuse_eval.py` (real backend). Importing any other `eval/` module — and therefore lint + the offline test suite — must require neither a Langfuse server nor the `langfuse` package for collection.
- Reuse existing metric code unchanged: `eval/retrieval_metrics.py`, `eval/generation_metrics.py`, `eval/llm_judge.py`, `eval/stats.py`, `eval/fast_subset.py`.
- Metrics are computed client-side and pushed as Langfuse scores. No server-side evaluators.
- Guardrails are forced OFF for eval pipeline builds (`enable_guardrails=False`), preserving current behavior.
- Test env: venv is pre-synced with `--extra all`. Run tests with `.venv/bin/python -m pytest`. Rely on exit code 0 (the final pytest summary line has a known shell-buffering quirk).
- Metric-name vocabulary (used verbatim as score names everywhere): `recall_at_5`, `precision_at_5`, `mrr`, `ndcg_at_5`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `judge_score`.

---

## File Structure

- `eval/langfuse_eval.py` (new) — normalized types (`GoldenItem`, `RunItemScores`), `Task`/`Evaluator` aliases, `EvalBackend` protocol, real `LangfuseEvalBackend`, `build_backend()`.
- `eval/evaluators.py` (new) — `build_evaluators(generator, embedder, *, judge_votes, judge_seed)` and the individual evaluator callables.
- `eval/experiment.py` (new, replaces `eval/run_eval.py`) — `build_task(pipeline)`, `run(...)` (dependency-injected), `main()` (argparse + real wiring).
- `eval/gate.py` (new, replaces `eval/compare.py`) — `evaluate_gate(...)`, `main()`.
- `eval/dataset_cli.py` (new) — `seed(...)`, `add_from_trace(...)`, `main()`.
- `tests/eval/fake_backend.py` (new) — `FakeEvalBackend` in-memory double.
- `tests/test_eval_backend.py`, `tests/test_eval_evaluators.py`, `tests/test_eval_experiment.py`, `tests/test_eval_gate.py`, `tests/test_eval_dataset_cli.py` (new).
- `core/config.py` (modify) — add `eval_gate_mode`, `eval_baseline_run`, `eval_gate_thresholds`.
- `eval/run_eval.py`, `eval/compare.py`, `tests/test_sp5_gate.py`, `tests/test_sp5_baseline_cli.py` (delete).
- `.github/workflows/eval-gate.yml`, `Makefile`, `docs/architecture.md`, `docs/PROJECT_STATUS.md` (modify).

---

## Task 1: Gate config knobs

**Files:**
- Modify: `core/config.py` (Settings class, near existing `eval_tolerance` at line ~151)
- Test: `tests/test_eval_config.py`

**Interfaces:**
- Produces: `Settings.eval_gate_mode: Literal["bootstrap","threshold","both"]` (default `"bootstrap"`), `Settings.eval_baseline_run: str` (default `"baseline"`), `Settings.eval_gate_thresholds: dict[str, float]` (default `{}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_config.py
from core import config


def test_eval_gate_defaults():
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.eval_gate_mode == "bootstrap"
    assert s.eval_baseline_run == "baseline"
    assert s.eval_gate_thresholds == {}


def test_eval_gate_mode_env_override(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("EVAL_GATE_MODE", "both")
    config.get_settings.cache_clear()
    assert config.get_settings().eval_gate_mode == "both"
    config.get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_config.py -v`
Expected: FAIL (`eval_gate_mode` attribute does not exist).

- [ ] **Step 3: Add the settings**

In `core/config.py`, immediately after the existing `eval_tolerance: float = 0.03` line, add:

```python
    eval_gate_mode: Literal["bootstrap", "threshold", "both"] = "bootstrap"
    eval_baseline_run: str = "baseline"
    eval_gate_thresholds: dict[str, float] = Field(default_factory=dict)
```

Ensure `Literal` is imported (it already is — used by `judge_provider`) and that `Field` is imported from pydantic (`from pydantic import Field`); add the import if missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_eval_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): add gate mode/baseline/threshold settings"
```

---

## Task 2: EvalBackend protocol, normalized types, and FakeEvalBackend

**Files:**
- Create: `eval/langfuse_eval.py`
- Create: `tests/eval/__init__.py` (empty)
- Create: `tests/eval/fake_backend.py`
- Test: `tests/test_eval_backend.py`

**Interfaces:**
- Produces:
  - `GoldenItem(id: str, question: str, expected_output: str = "", relevant_chunk_ids: list[str] = [], tenant_id: str = "public")` (dataclass).
  - `RunItemScores(item_id: str, scores: dict[str, float])` (dataclass).
  - `Task = Callable[[GoldenItem], dict]` — returns a pipeline-output dict.
  - `Evaluator = Callable[[GoldenItem, dict], list[tuple[str, float]]]` — `(item, output) -> [(score_name, value), ...]`.
  - `class EvalBackend(Protocol)` with methods:
    - `ensure_dataset(self, name: str) -> None`
    - `upsert_item(self, *, dataset: str, item: GoldenItem) -> None`
    - `get_dataset_items(self, name: str) -> list[GoldenItem]`
    - `add_item_from_trace(self, *, dataset: str, trace_id: str) -> None`
    - `run_experiment(self, *, dataset: str, run_name: str, items: list[GoldenItem], task: Task, evaluators: list[Evaluator], max_concurrency: int = 8) -> None`
    - `get_run_scores(self, *, dataset: str, run_name: str) -> list[RunItemScores]`
  - `FakeEvalBackend` (in `tests/eval/fake_backend.py`) implementing the protocol in memory.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_backend.py
from eval.langfuse_eval import GoldenItem, RunItemScores
from tests.eval.fake_backend import FakeEvalBackend


def _item(i):
    return GoldenItem(id=i, question=f"q{i}", expected_output=f"a{i}",
                      relevant_chunk_ids=[f"c{i}"], tenant_id="public")


def test_seed_and_get_items_roundtrip():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.upsert_item(dataset="ds", item=_item("1"))
    be.upsert_item(dataset="ds", item=_item("1"))  # idempotent by id
    be.upsert_item(dataset="ds", item=_item("2"))
    items = be.get_dataset_items("ds")
    assert sorted(i.id for i in items) == ["1", "2"]
    assert items[0].question == "q1"


def test_run_experiment_executes_task_and_evaluators():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.upsert_item(dataset="ds", item=_item("1"))

    def task(item):
        return {"answer": item.question.upper()}

    def evaluator(item, output):
        return [("len", float(len(output["answer"])))]

    be.run_experiment(dataset="ds", run_name="r1",
                      items=be.get_dataset_items("ds"),
                      task=task, evaluators=[evaluator])
    scores = be.get_run_scores(dataset="ds", run_name="r1")
    assert scores == [RunItemScores(item_id="1", scores={"len": 2.0})]


def test_add_item_from_trace_records_source():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.add_item_from_trace(dataset="ds", trace_id="tr-9")
    assert be.trace_items["ds"] == ["tr-9"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_backend.py -v`
Expected: FAIL (`eval.langfuse_eval` / `tests.eval.fake_backend` do not exist).

- [ ] **Step 3: Create `eval/langfuse_eval.py`**

```python
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
```

- [ ] **Step 4: Create `tests/eval/__init__.py` and `tests/eval/fake_backend.py`**

`tests/eval/__init__.py`: empty file.

```python
# tests/eval/fake_backend.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_backend.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add eval/langfuse_eval.py tests/eval/__init__.py tests/eval/fake_backend.py tests/test_eval_backend.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): EvalBackend protocol + normalized types + in-memory fake"
```

---

## Task 3: Evaluator adapters

**Files:**
- Create: `eval/evaluators.py`
- Test: `tests/test_eval_evaluators.py`

**Interfaces:**
- Consumes: `GoldenItem`, `Evaluator` (Task 2); existing `eval.retrieval_metrics` (`recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`), `eval.generation_metrics` (`faithfulness(question, answer, contexts, generator)`, `answer_relevancy(question, answer, generator, embedder)`, `context_precision(question, answer, contexts, generator)`, `context_recall(question, ground_truth, contexts, generator)`), `eval.llm_judge.holistic_judge(question, answer, contexts, generator, votes, base_seed)`.
- Produces:
  - `retrieval_evaluator(item: GoldenItem, output: dict) -> list[tuple[str, float]]` — returns `recall_at_5`, `precision_at_5`, `mrr`, `ndcg_at_5`.
  - `build_evaluators(generator, embedder, *, judge_votes: int = 1, judge_seed: int = 0) -> list[Evaluator]` — retrieval + generation + judge evaluators.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_evaluators.py
from eval.evaluators import build_evaluators, retrieval_evaluator
from eval.langfuse_eval import GoldenItem


def test_retrieval_evaluator_exact_values():
    item = GoldenItem(id="1", question="q", relevant_chunk_ids=["c1", "c2"])
    output = {"retrieved_ids": ["c1", "x", "c2", "y", "z"]}
    scores = dict(retrieval_evaluator(item, output))
    assert scores["recall_at_5"] == 1.0
    assert scores["precision_at_5"] == 2 / 5
    assert scores["mrr"] == 1.0
    assert 0.0 < scores["ndcg_at_5"] <= 1.0


class _StubGen:
    """Generator stub: holistic_judge + generation metrics call .complete()."""

    model = "stub-gen"

    def complete(self, messages, response_model=None, **kwargs):
        class _Resp:
            parsed = {"score": 0.5, "rationale": "ok"}
        return _Resp()


class _StubEmbed:
    model = "stub-embed"

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_build_evaluators_emits_all_score_names():
    item = GoldenItem(id="1", question="q", expected_output="gt",
                      relevant_chunk_ids=["c1"])
    output = {"retrieved_ids": ["c1"], "answer": "a", "contexts": ["ctx"]}
    evaluators = build_evaluators(_StubGen(), _StubEmbed(), judge_votes=1)
    names = set()
    for ev in evaluators:
        for name, value in ev(item, output):
            names.add(name)
            assert isinstance(value, float)
    assert names == {
        "recall_at_5", "precision_at_5", "mrr", "ndcg_at_5",
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "judge_score",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_evaluators.py -v`
Expected: FAIL (`eval.evaluators` does not exist).

- [ ] **Step 3: Create `eval/evaluators.py`**

```python
"""Adapters wrapping the existing metric functions as EvalBackend evaluators.

Each evaluator has signature (GoldenItem, output_dict) -> [(score_name, value)].
Metric names are fixed (see the plan's metric-name vocabulary).
"""

from __future__ import annotations

from eval.generation_metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from eval.langfuse_eval import Evaluator, GoldenItem
from eval.llm_judge import holistic_judge
from eval.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def retrieval_evaluator(item: GoldenItem, output: dict) -> list[tuple[str, float]]:
    retrieved = output.get("retrieved_ids", [])
    relevant = set(item.relevant_chunk_ids)
    return [
        ("recall_at_5", recall_at_k(retrieved, relevant, 5)),
        ("precision_at_5", precision_at_k(retrieved, relevant, 5)),
        ("mrr", mrr(retrieved, relevant)),
        ("ndcg_at_5", ndcg_at_k(retrieved, relevant, 5)),
    ]


def build_evaluators(generator, embedder, *, judge_votes: int = 1,
                     judge_seed: int = 0) -> list[Evaluator]:
    def generation_evaluator(item: GoldenItem, output: dict) -> list[tuple[str, float]]:
        q = item.question
        answer = output.get("answer", "")
        contexts = output.get("contexts", [])
        return [
            ("faithfulness", faithfulness(q, answer, contexts, generator)),
            ("answer_relevancy", answer_relevancy(q, answer, generator, embedder)),
            ("context_precision", context_precision(q, answer, contexts, generator)),
            ("context_recall", context_recall(q, item.expected_output, contexts, generator)),
        ]

    def judge_evaluator(item: GoldenItem, output: dict) -> list[tuple[str, float]]:
        result = holistic_judge(
            item.question, output.get("answer", ""), output.get("contexts", []),
            generator, votes=judge_votes, base_seed=judge_seed,
        )
        return [("judge_score", float(result["score"]))]

    return [retrieval_evaluator, generation_evaluator, judge_evaluator]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_evaluators.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/evaluators.py tests/test_eval_evaluators.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): evaluator adapters over existing metric functions"
```

---

## Task 4: Experiment runner

**Files:**
- Create: `eval/experiment.py`
- Test: `tests/test_eval_experiment.py`

**Interfaces:**
- Consumes: `EvalBackend`, `GoldenItem`, `Task` (Task 2); `build_evaluators` (Task 3); `eval.fast_subset.fast_subset(items, n, seed)`; `core.registry.build_generator`, `core.registry.build_embedder`; `core.pipeline.build`.
- Produces:
  - `build_task(pipeline) -> Task` — runs the pipeline per item within the item's tenant ACL.
  - `run(*, backend, pipeline, generator, embedder, dataset, run_name, limit=None, fast=False, judge_votes=1, judge_seed=0, max_concurrency=8) -> str` — subsets items, submits the experiment, returns `run_name`.
  - `main()` — argparse entry point wiring real backend/pipeline/models.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_experiment.py
from eval.experiment import build_task, run
from eval.langfuse_eval import GoldenItem
from tests.eval.fake_backend import FakeEvalBackend


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, question, acl=None, *, collection_id=None):
        self.calls.append((question, getattr(acl, "tenant_id", None)))
        return {"answer": question.upper(), "retrieved_ids": ["c1"], "contexts": ["ctx"]}


def test_build_task_runs_pipeline_within_tenant():
    pipe = _FakePipeline()
    task = build_task(pipe)
    out = task(GoldenItem(id="1", question="hi", tenant_id="acme"))
    assert out["answer"] == "HI"
    assert pipe.calls == [("hi", "acme")]


def _stub_models():
    class _Gen:
        model = "g"
        def complete(self, messages, response_model=None, **kw):
            class _R: parsed = {"score": 0.5, "rationale": ""}
            return _R()
    class _Emb:
        model = "e"
        def embed_query(self, t): return [1.0, 0.0]
        def embed_documents(self, ts): return [[1.0, 0.0] for _ in ts]
    return _Gen(), _Emb()


def test_run_submits_experiment_with_scores():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.upsert_item(dataset="ds", item=GoldenItem(id="1", question="q1",
                   expected_output="gt", relevant_chunk_ids=["c1"]))
    gen, emb = _stub_models()
    name = run(backend=be, pipeline=_FakePipeline(), generator=gen, embedder=emb,
               dataset="ds", run_name="r1", judge_votes=1)
    assert name == "r1"
    scores = be.get_run_scores(dataset="ds", run_name="r1")[0].scores
    assert scores["recall_at_5"] == 1.0
    assert "judge_score" in scores


def test_run_fast_subsets_items():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    for i in range(20):
        be.upsert_item(dataset="ds", item=GoldenItem(id=str(i), question=f"q{i}"))
    gen, emb = _stub_models()
    run(backend=be, pipeline=_FakePipeline(), generator=gen, embedder=emb,
        dataset="ds", run_name="r1", fast=True, judge_votes=1)
    assert len(be.get_run_scores(dataset="ds", run_name="r1")) == 15  # eval_fast_n default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_experiment.py -v`
Expected: FAIL (`eval.experiment` does not exist).

- [ ] **Step 3: Create `eval/experiment.py`**

```python
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
                              enable_guardrails=False)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_experiment.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/experiment.py tests/test_eval_experiment.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): Langfuse-native experiment runner"
```

---

## Task 5: Regression gate

**Files:**
- Create: `eval/gate.py`
- Test: `tests/test_eval_gate.py`

**Interfaces:**
- Consumes: `EvalBackend`, `RunItemScores` (Task 2); `eval.stats.paired_bootstrap(a, b, n, alpha, seed)`; `Settings.eval_gate_mode`, `Settings.eval_baseline_run`, `Settings.eval_gate_thresholds`, `Settings.eval_tolerance`, `Settings.eval_bootstrap_resamples` (Task 1 + existing).
- Produces:
  - `evaluate_gate(*, backend, dataset, new_run, baseline_run, mode="bootstrap", tolerance=0.03, thresholds=None, resamples=1000) -> bool` — True iff the gate passes. Prints a table; raises `ValueError` on structural problems (unknown run, item-set mismatch, missing metric on one side).
  - `main()` — argparse entry point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_gate.py
import pytest

from eval.gate import evaluate_gate
from eval.langfuse_eval import RunItemScores
from tests.eval.fake_backend import FakeEvalBackend


def _seed_runs(base_vals, new_vals, metric="recall_at_5"):
    be = FakeEvalBackend()
    be.runs[("ds", "baseline")] = [
        RunItemScores(item_id=str(i), scores={metric: v}) for i, v in enumerate(base_vals)
    ]
    be.runs[("ds", "new")] = [
        RunItemScores(item_id=str(i), scores={metric: v}) for i, v in enumerate(new_vals)
    ]
    return be


def test_bootstrap_passes_when_equal():
    be = _seed_runs([0.8] * 10, [0.8] * 10)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="bootstrap",
                         tolerance=0.03, resamples=200) is True


def test_bootstrap_fails_on_regression():
    be = _seed_runs([0.9] * 10, [0.5] * 10)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="bootstrap",
                         tolerance=0.03, resamples=200) is False


def test_threshold_fails_below_floor():
    be = _seed_runs([0.9] * 5, [0.6] * 5)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="threshold",
                         thresholds={"recall_at_5": 0.7}, resamples=200) is False


def test_threshold_passes_above_floor():
    be = _seed_runs([0.9] * 5, [0.8] * 5)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="threshold",
                         thresholds={"recall_at_5": 0.7}, resamples=200) is True


def test_both_fails_if_either_fails():
    be = _seed_runs([0.9] * 8, [0.85] * 8)  # within tolerance, but below floor 0.9
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="both",
                         tolerance=0.03, thresholds={"recall_at_5": 0.9},
                         resamples=200) is False


def test_item_set_mismatch_raises():
    be = FakeEvalBackend()
    be.runs[("ds", "baseline")] = [RunItemScores(item_id="0", scores={"m": 0.5})]
    be.runs[("ds", "new")] = [RunItemScores(item_id="1", scores={"m": 0.5})]
    with pytest.raises(ValueError):
        evaluate_gate(backend=be, dataset="ds", new_run="new",
                      baseline_run="baseline", mode="bootstrap", resamples=50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_gate.py -v`
Expected: FAIL (`eval.gate` does not exist).

- [ ] **Step 3: Create `eval/gate.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_gate.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/gate.py tests/test_eval_gate.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): configurable bootstrap/threshold regression gate"
```

---

## Task 6: Curation CLI (seed + add-from-trace)

**Files:**
- Create: `eval/dataset_cli.py`
- Test: `tests/test_eval_dataset_cli.py`

**Interfaces:**
- Consumes: `EvalBackend`, `GoldenItem` (Task 2).
- Produces:
  - `load_items(path: str) -> list[GoldenItem]` — parse a JSON array of `{id, question, ground_truth, relevant_chunk_ids, tenant_id}`.
  - `seed(*, backend, dataset: str, items: list[GoldenItem]) -> int` — ensure dataset, upsert each item, return count.
  - `add_from_trace(*, backend, dataset: str, trace_id: str) -> None`.
  - `main()` — argparse with `seed` and `add-from-trace` subcommands.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_dataset_cli.py
import json

from eval.dataset_cli import add_from_trace, load_items, seed
from tests.eval.fake_backend import FakeEvalBackend


def test_load_items_parses_fields(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([
        {"id": "1", "question": "q1", "ground_truth": "a1",
         "relevant_chunk_ids": ["c1"], "tenant_id": "public"},
    ]))
    items = load_items(str(p))
    assert items[0].id == "1"
    assert items[0].expected_output == "a1"
    assert items[0].relevant_chunk_ids == ["c1"]


def test_seed_is_idempotent(tmp_path):
    be = FakeEvalBackend()
    items = load_items(str(_write(tmp_path)))
    assert seed(backend=be, dataset="ds", items=items) == 1
    seed(backend=be, dataset="ds", items=items)  # again
    assert len(be.get_dataset_items("ds")) == 1


def test_add_from_trace_records(tmp_path):
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    add_from_trace(backend=be, dataset="ds", trace_id="tr-1")
    assert be.trace_items["ds"] == ["tr-1"]


def _write(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([{"id": "1", "question": "q1", "ground_truth": "a1"}]))
    return p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_dataset_cli.py -v`
Expected: FAIL (`eval.dataset_cli` does not exist).

- [ ] **Step 3: Create `eval/dataset_cli.py`**

```python
"""Dataset curation CLI: seed a dataset from a local file, or promote a trace.

`seed` bootstraps/updates a Langfuse dataset from a committed JSON file (also used
by tests). `add-from-trace` promotes a production trace to a dataset item; the
expected output is filled in later by a human in the Langfuse UI.
"""

from __future__ import annotations

import argparse
import json

from eval.langfuse_eval import EvalBackend, GoldenItem, build_backend


def load_items(path: str) -> list[GoldenItem]:
    with open(path) as f:
        raw = json.load(f)
    return [
        GoldenItem(
            id=str(r["id"]),
            question=r["question"],
            expected_output=r.get("ground_truth", r.get("expected_output", "")),
            relevant_chunk_ids=list(r.get("relevant_chunk_ids", [])),
            tenant_id=r.get("tenant_id", "public"),
        )
        for r in raw
    ]


def seed(*, backend: EvalBackend, dataset: str, items: list[GoldenItem]) -> int:
    backend.ensure_dataset(dataset)
    for item in items:
        backend.upsert_item(dataset=dataset, item=item)
    return len(items)


def add_from_trace(*, backend: EvalBackend, dataset: str, trace_id: str) -> None:
    backend.ensure_dataset(dataset)
    backend.add_item_from_trace(dataset=dataset, trace_id=trace_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse dataset curation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Seed a dataset from a local JSON file.")
    p_seed.add_argument("--dataset", required=True)
    p_seed.add_argument("--items", required=True, help="Path to a JSON array of items.")

    p_trace = sub.add_parser("add-from-trace", help="Promote a trace to a dataset item.")
    p_trace.add_argument("--dataset", required=True)
    p_trace.add_argument("--trace-id", required=True)

    args = parser.parse_args()
    backend = build_backend()

    if args.command == "seed":
        count = seed(backend=backend, dataset=args.dataset,
                     items=load_items(args.items))
        print(f"[dataset] Seeded {count} items into '{args.dataset}'")
    elif args.command == "add-from-trace":
        add_from_trace(backend=backend, dataset=args.dataset, trace_id=args.trace_id)
        print(f"[dataset] Added item from trace '{args.trace_id}' to '{args.dataset}'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_dataset_cli.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/dataset_cli.py tests/test_eval_dataset_cli.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): dataset curation CLI (seed + add-from-trace)"
```

---

## Task 7: Real Langfuse backend

**Files:**
- Create: `eval/_langfuse_backend.py`
- Test: `tests/test_eval_backend_lazy.py`

**Interfaces:**
- Consumes: `GoldenItem`, `RunItemScores`, `Task`, `Evaluator` (Task 2). Langfuse SDK 4.9.1: `Langfuse(host, public_key, secret_key)`, `get_dataset(name).items` (each `.id`, `.input`, `.expected_output`, `.metadata`), `create_dataset(name=)`, `create_dataset_item(dataset_name=, id=, input=, expected_output=, metadata=, source_trace_id=)`, `run_experiment(name=, run_name=, data=, task=, evaluators=)`, `get_dataset_run(dataset_name=, run_name=)` (`.id`, `.dataset_run_items[*].trace_id`, `.dataset_run_items[*].dataset_item_id`), `api.scores.get_many(dataset_run_id=, page=, limit=)` (`.data[*].name/.value/.trace_id`), `langfuse.experiment.Evaluation(name=, value=)`.
- Produces: `LangfuseEvalBackend(settings=None)` implementing `EvalBackend`.

**Note:** This module's live behavior is verified by the CI eval job, not by offline
unit tests. The offline test here only asserts that importing sibling `eval` modules
does not import `langfuse`, and that construction fails cleanly without a server. Keep
all `langfuse` imports inside methods/`__init__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_backend_lazy.py
import sys


def test_importing_eval_modules_does_not_import_langfuse():
    for m in list(sys.modules):
        if m == "langfuse" or m.startswith("langfuse."):
            del sys.modules[m]
    import importlib
    for name in ("eval.langfuse_eval", "eval.experiment", "eval.gate",
                 "eval.evaluators", "eval.dataset_cli"):
        importlib.import_module(name)
    assert "langfuse" not in sys.modules, "eval modules must not import langfuse at import time"


def test_real_backend_class_exists():
    from eval._langfuse_backend import LangfuseEvalBackend
    assert hasattr(LangfuseEvalBackend, "run_experiment")
    assert hasattr(LangfuseEvalBackend, "get_run_scores")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_backend_lazy.py -v`
Expected: FAIL (`eval._langfuse_backend` does not exist).

- [ ] **Step 3: Create `eval/_langfuse_backend.py`**

```python
"""Real Langfuse-backed EvalBackend. The ONLY module importing `langfuse`.

Live behavior is verified by the CI eval job; offline tests use FakeEvalBackend.
"""

from __future__ import annotations

from eval.langfuse_eval import Evaluator, GoldenItem, RunItemScores, Task


class LangfuseEvalBackend:
    def __init__(self, settings=None) -> None:
        from langfuse import Langfuse

        from core.config import get_settings

        s = settings or get_settings()
        self._client = Langfuse(
            host=s.langfuse_host,
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
        )

    def ensure_dataset(self, name: str) -> None:
        try:
            self._client.get_dataset(name)
        except Exception:
            self._client.create_dataset(name=name)

    def upsert_item(self, *, dataset: str, item: GoldenItem) -> None:
        # Deterministic id makes re-seeding idempotent (create acts as upsert on id).
        self._client.create_dataset_item(
            dataset_name=dataset,
            id=item.id,
            input={"question": item.question},
            expected_output=item.expected_output,
            metadata={
                "item_id": item.id,
                "relevant_chunk_ids": item.relevant_chunk_ids,
                "tenant_id": item.tenant_id,
            },
        )

    def get_dataset_items(self, name: str) -> list[GoldenItem]:
        dataset = self._client.get_dataset(name)
        items: list[GoldenItem] = []
        for it in dataset.items:
            meta = it.metadata or {}
            inp = it.input or {}
            question = inp.get("question", "") if isinstance(inp, dict) else str(inp)
            items.append(GoldenItem(
                id=str(getattr(it, "id", meta.get("item_id", ""))),
                question=question,
                expected_output=it.expected_output or "",
                relevant_chunk_ids=list(meta.get("relevant_chunk_ids", [])),
                tenant_id=meta.get("tenant_id", "public"),
            ))
        return items

    def add_item_from_trace(self, *, dataset: str, trace_id: str) -> None:
        # Provenance is native via source_trace_id; a human fills expected_output
        # in the Langfuse UI. input left empty here (linked trace shows the question).
        self._client.create_dataset_item(
            dataset_name=dataset,
            source_trace_id=trace_id,
            input={},
            expected_output="",
            metadata={"tenant_id": "public"},
        )

    def run_experiment(self, *, dataset: str, run_name: str, items: list[GoldenItem],
                       task: Task, evaluators: list[Evaluator],
                       max_concurrency: int = 8) -> None:
        from langfuse.experiment import Evaluation

        wanted_ids = {i.id for i in items}
        sdk_dataset = self._client.get_dataset(dataset)
        data = [it for it in sdk_dataset.items
                if str(getattr(it, "id", (it.metadata or {}).get("item_id"))) in wanted_ids]

        def _golden(sdk_item) -> GoldenItem:
            meta = sdk_item.metadata or {}
            inp = sdk_item.input or {}
            q = inp.get("question", "") if isinstance(inp, dict) else str(inp)
            return GoldenItem(
                id=str(getattr(sdk_item, "id", meta.get("item_id", ""))),
                question=q,
                expected_output=sdk_item.expected_output or "",
                relevant_chunk_ids=list(meta.get("relevant_chunk_ids", [])),
                tenant_id=meta.get("tenant_id", "public"),
            )

        def _task(*, item, **_kw):
            return task(_golden(item))

        def _evaluator(*, input, output, expected_output, metadata, **_kw):
            golden = GoldenItem(
                id=str((metadata or {}).get("item_id", "")),
                question=(input or {}).get("question", "") if isinstance(input, dict) else str(input),
                expected_output=expected_output or "",
                relevant_chunk_ids=list((metadata or {}).get("relevant_chunk_ids", [])),
                tenant_id=(metadata or {}).get("tenant_id", "public"),
            )
            evals = []
            for ev in evaluators:
                for name, value in ev(golden, output):
                    evals.append(Evaluation(name=name, value=float(value)))
            return evals

        self._client.run_experiment(
            name=dataset, run_name=run_name, data=data,
            task=_task, evaluators=[_evaluator], max_concurrency=max_concurrency,
        )
        self._client.flush()

    def get_run_scores(self, *, dataset: str, run_name: str) -> list[RunItemScores]:
        run = self._client.get_dataset_run(dataset_name=dataset, run_name=run_name)
        trace_to_item = {ri.trace_id: ri.dataset_item_id for ri in run.dataset_run_items}

        # Fetch every score attached to this run, paginating.
        by_trace: dict[str, dict[str, float]] = {}
        page = 1
        while True:
            resp = self._client.api.scores.get_many(
                dataset_run_id=run.id, page=page, limit=100)
            data = getattr(resp, "data", []) or []
            for sc in data:
                if sc.trace_id is None or not isinstance(sc.value, (int, float)):
                    continue
                by_trace.setdefault(sc.trace_id, {})[sc.name] = float(sc.value)
            if len(data) < 100:
                break
            page += 1

        results: list[RunItemScores] = []
        for trace_id, item_id in trace_to_item.items():
            results.append(RunItemScores(item_id=str(item_id),
                                         scores=by_trace.get(trace_id, {})))
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_backend_lazy.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/_langfuse_backend.py tests/test_eval_backend_lazy.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): real Langfuse-backed EvalBackend (lazy-imported)"
```

---

## Task 8: Remove old eval modules, wire CI + Makefile + docs

**Files:**
- Delete: `eval/run_eval.py`, `eval/compare.py`, `tests/test_sp5_gate.py`, `tests/test_sp5_baseline_cli.py`
- Modify: `.github/workflows/eval-gate.yml`
- Modify: `Makefile`
- Modify: `docs/architecture.md`, `docs/PROJECT_STATUS.md`

**Interfaces:**
- Consumes: `eval.experiment`, `eval.gate`, `eval.dataset_cli` (Tasks 4–6).

- [ ] **Step 1: Delete superseded modules and their tests**

```bash
git rm eval/run_eval.py eval/compare.py tests/test_sp5_gate.py tests/test_sp5_baseline_cli.py
```

- [ ] **Step 2: Verify nothing else imports them**

Run: `grep -rn "run_eval\|eval.compare\|eval import compare" eval/ tests/ app/ core/ ingest/ retrieval/`
Expected: no matches (docs/historical plans may still mention them — those are fine).

- [ ] **Step 3: Update the CI eval job**

In `.github/workflows/eval-gate.yml`, update the `eval` job. (a) Add Langfuse secrets to its `env:` block:

```yaml
      LANGFUSE_ENABLED: "true"
      LANGFUSE_HOST: ${{ secrets.LANGFUSE_HOST }}
      LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
      LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
```

(b) Fix the stale embedding pin — change `EMBED_MODEL: nvidia/nv-embedqa-e5-v5` to `EMBED_MODEL: baai/bge-m3`.

(c) Replace the "Run eval" and "Compare against baseline" steps with:

```yaml
      - name: Run eval experiment (full pipeline, fast subset)
        run: |
          uv run python -m eval.experiment \
            --dataset hotpotqa \
            --version full \
            --fast \
            --run-name "ci-${{ github.run_id }}"

      - name: Gate against baseline run
        run: |
          uv run python -m eval.gate \
            --dataset hotpotqa \
            --new-run "ci-${{ github.run_id }}" \
            --baseline-run baseline
```

Leave the `lint` and `acl-isolation` jobs unchanged.

- [ ] **Step 4: Update the Makefile**

Replace the `eval`, `compare`, and baseline targets. Change the `.PHONY` line's `eval compare` to `eval gate seed`, then:

```makefile
eval:               ## Run eval experiment:  make eval DATASET=hotpotqa RUN=myrun
	uv run python -m eval.experiment --dataset $(DATASET) --version full --run-name $(RUN)

gate:               ## Gate a run vs baseline:  make gate DATASET=hotpotqa RUN=myrun
	uv run python -m eval.gate --dataset $(DATASET) --new-run $(RUN) --baseline-run baseline

seed:               ## Seed a dataset:  make seed DATASET=hotpotqa ITEMS=path.json
	uv run python -m eval.dataset_cli seed --dataset $(DATASET) --items $(ITEMS)
```

Remove the old baseline-writing target that referenced `eval.run_eval --write-baseline`, and drop `eval/runs` from the `clean` target's `rm -rf` list (no longer produced).

- [ ] **Step 5: Update docs**

In `docs/architecture.md` and `docs/PROJECT_STATUS.md`, update the eval section: datasets/runs live in Langfuse; the runner is `eval.experiment` (Langfuse `run_experiment`), the gate is `eval.gate` (paired-bootstrap and/or thresholds, reading scores back from Langfuse), curation via `eval.dataset_cli`. Note the CI eval job now requires `LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` secrets. Remove references to `eval/baselines/*.json`, `eval/run_eval.py`, and `eval/compare.py`.

- [ ] **Step 6: Run the full offline suite**

Run: `.venv/bin/python -m pytest -q`
Expected: exit code 0 (green; ~live-backend skips only). No collection errors from the deleted modules.

- [ ] **Step 7: Lint**

Run: `.venv/bin/python -m ruff check eval/ tests/`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "refactor(eval): remove file-based runner/gate; wire Langfuse eval into CI, Makefile, docs"
```

---

## Self-Review

**Spec coverage:**
- Experiment tracking → Tasks 4 + 7 (`run_experiment`, linked traces, scores). ✓
- Trace-curation → Task 6 (`add-from-trace`) + Task 7 (`source_trace_id`). ✓
- Langfuse-native (Approach B) → Task 7 uses `run_experiment`. ✓
- Client-side metrics pushed as scores → Task 3 evaluators + Task 7 `Evaluation`. ✓
- Configurable gate (bootstrap/threshold/both) → Tasks 1 + 5. ✓
- EvalBackend seam, langfuse lazy-imported, offline-testable → Tasks 2 + 7 (lazy-import test). ✓
- Remove `run_eval.py`/`compare.py` (no compat shim) → Task 8. ✓
- CI wiring + stale e5-v5→bge-m3 fix → Task 8. ✓
- Scope: plumbing only, no live baseline → no task produces a dataset/baseline. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test has assertions. ✓

**Type consistency:** `GoldenItem`, `RunItemScores`, `Task`, `Evaluator`, `EvalBackend` method names identical across Tasks 2–7. Score names match the Global Constraints vocabulary in Tasks 3/5/7. `run(...)`/`evaluate_gate(...)`/`seed(...)` signatures consistent between definition and callers. ✓
