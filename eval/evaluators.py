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
