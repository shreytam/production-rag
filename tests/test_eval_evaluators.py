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
