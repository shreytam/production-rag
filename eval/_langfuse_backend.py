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
