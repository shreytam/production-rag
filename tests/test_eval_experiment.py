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
            class _R:
                parsed = {"score": 0.5, "rationale": ""}
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
