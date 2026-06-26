"""Offline tests for eval metrics.

All LLM / embedder calls use injected fakes — no network required.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import pytest

from core.types import ChatMessage, LLMResponse, Usage
from eval.fast_subset import fast_subset
from eval.generation_metrics import answer_relevancy, context_precision, context_recall, faithfulness
from eval.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from eval.stats import bootstrap_ci, paired_bootstrap


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGenerator:
    """Generator that pops canned LLMResponse objects from a queue."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._queue: deque[dict] = deque(responses)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self._queue:
            raise RuntimeError("FakeGenerator: no more canned responses")
        parsed = self._queue.popleft()
        return LLMResponse(text="", parsed=parsed, usage=Usage(), model="fake")


class FakeEmbedder:
    """Embedder returning canned vectors.

    ``query_vec`` is returned for ``embed_query``.
    ``doc_vecs`` is cycled for each ``embed_documents`` call.
    """

    def __init__(self, query_vec: list[float], doc_vecs: list[list[float]]) -> None:
        self._query_vec = query_vec
        self._doc_vecs = doc_vecs

    @property
    def dimension(self) -> int:
        return len(self._query_vec)

    def embed_query(self, text: str) -> list[float]:
        return self._query_vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._doc_vecs[: len(texts)]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    """retrieved=["d1","d2","d3"], relevant={"d2"}"""

    retrieved = ["d1", "d2", "d3"]
    relevant = {"d2"}

    def test_recall_at_3(self) -> None:
        assert recall_at_k(self.retrieved, self.relevant, 3) == pytest.approx(1.0)

    def test_precision_at_3(self) -> None:
        assert precision_at_k(self.retrieved, self.relevant, 3) == pytest.approx(1 / 3)

    def test_mrr(self) -> None:
        # "d2" is at rank 2 → reciprocal rank = 1/2
        assert mrr(self.retrieved, self.relevant) == pytest.approx(0.5)

    def test_ndcg_at_3(self) -> None:
        # DCG  = 1/log2(3)   (hit at rank 2)
        # IDCG = 1/log2(2) = 1  (ideal: hit at rank 1)
        # nDCG = DCG / IDCG = 1/log2(3)
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k(self.retrieved, self.relevant, 3) == pytest.approx(expected)

    def test_perfect_rank(self) -> None:
        """Relevant doc at rank 1 → all metrics should equal 1.0."""
        perfect = ["d2", "d1", "d3"]
        assert recall_at_k(perfect, self.relevant, 3) == pytest.approx(1.0)
        assert precision_at_k(perfect, self.relevant, 1) == pytest.approx(1.0)
        assert mrr(perfect, self.relevant) == pytest.approx(1.0)
        assert ndcg_at_k(perfect, self.relevant, 3) == pytest.approx(1.0)

    def test_no_hit(self) -> None:
        assert recall_at_k(["d1", "d3"], self.relevant, 2) == pytest.approx(0.0)
        assert mrr(["d1", "d3"], self.relevant) == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        """recall vacuously 1.0 when relevant set is empty."""
        assert recall_at_k(self.retrieved, set(), 3) == pytest.approx(1.0)

    def test_k_zero(self) -> None:
        assert recall_at_k(self.retrieved, self.relevant, 0) == pytest.approx(0.0)
        assert precision_at_k(self.retrieved, self.relevant, 0) == pytest.approx(0.0)
        assert ndcg_at_k(self.retrieved, self.relevant, 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_bootstrap_ci_constant(self) -> None:
        mean, lo, hi = bootstrap_ci([0.5] * 100)
        assert mean == pytest.approx(0.5)
        assert lo <= 0.5 <= hi

    def test_bootstrap_ci_deterministic(self) -> None:
        r1 = bootstrap_ci([0.1, 0.5, 0.9, 0.3], seed=42)
        r2 = bootstrap_ci([0.1, 0.5, 0.9, 0.3], seed=42)
        assert r1 == r2

    def test_bootstrap_ci_different_seeds(self) -> None:
        r1 = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7], seed=0)
        r2 = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7], seed=1)
        # CIs may differ between seeds (probabilistically almost certain)
        # Just ensure they are valid intervals
        for mean, lo, hi in (r1, r2):
            assert lo <= mean <= hi

    def test_paired_bootstrap(self) -> None:
        a = [0.4, 0.5, 0.6]
        b = [0.5, 0.6, 0.7]
        diff, lo, hi = paired_bootstrap(a, b, seed=0)
        assert diff == pytest.approx(0.1)
        assert lo <= diff <= hi

    def test_paired_bootstrap_deterministic(self) -> None:
        a = [0.1, 0.2, 0.3]
        b = [0.4, 0.5, 0.6]
        assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)

    def test_paired_bootstrap_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap([1.0, 2.0], [1.0], seed=0)


# ---------------------------------------------------------------------------
# Generation metrics (offline with fake generator / embedder)
# ---------------------------------------------------------------------------


class TestFaithfulness:
    """Test faithfulness with a FakeGenerator."""

    contexts = ["The Eiffel Tower is 330 m tall.", "It was completed in 1889."]

    def test_all_supported(self) -> None:
        """2 claims, both supported → score == 1.0."""
        gen = FakeGenerator(
            [
                # call 1: extract claims
                {"claims": ["The Eiffel Tower is 330 m tall.", "It was completed in 1889."]},
                # call 2: judge verdicts – both supported
                {
                    "verdicts": [
                        {"claim": "The Eiffel Tower is 330 m tall.", "supported": True},
                        {"claim": "It was completed in 1889.", "supported": True},
                    ]
                },
            ]
        )
        score = faithfulness(
            question="Tell me about the Eiffel Tower.",
            answer="The Eiffel Tower is 330 m tall and was completed in 1889.",
            contexts=self.contexts,
            generator=gen,
        )
        assert score == pytest.approx(1.0)

    def test_half_supported(self) -> None:
        """2 claims, 1 supported → score == 0.5."""
        gen = FakeGenerator(
            [
                {"claims": ["Claim A", "Claim B"]},
                {
                    "verdicts": [
                        {"claim": "Claim A", "supported": True},
                        {"claim": "Claim B", "supported": False},
                    ]
                },
            ]
        )
        score = faithfulness(
            question="Q",
            answer="Claim A. Claim B.",
            contexts=self.contexts,
            generator=gen,
        )
        assert score == pytest.approx(0.5)

    def test_no_claims(self) -> None:
        """Empty claim list → 0.0."""
        gen = FakeGenerator([{"claims": []}])
        score = faithfulness("Q", "A", self.contexts, gen)
        assert score == pytest.approx(0.0)


class TestAnswerRelevancy:
    """Test answer_relevancy with fake generator + embedder."""

    def test_identical_vectors(self) -> None:
        """When all generated-question vectors == query vector, similarity = 1.0."""
        query_vec = [1.0, 0.0]
        # 3 doc vecs identical to the query vec
        doc_vecs = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

        gen = FakeGenerator(
            [{"questions": ["Q1?", "Q2?", "Q3?"]}]
        )
        emb = FakeEmbedder(query_vec=query_vec, doc_vecs=doc_vecs)

        score = answer_relevancy(
            question="What is X?",
            answer="X is Y.",
            generator=gen,
            embedder=emb,
            n_questions=3,
        )
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """When generated-question vectors are orthogonal to query, similarity = 0.0."""
        query_vec = [1.0, 0.0]
        doc_vecs = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

        gen = FakeGenerator([{"questions": ["Q1?", "Q2?", "Q3?"]}])
        emb = FakeEmbedder(query_vec=query_vec, doc_vecs=doc_vecs)

        score = answer_relevancy("What is X?", "X is Y.", gen, emb)
        assert score == pytest.approx(0.0)

    def test_partial_similarity(self) -> None:
        """Mean cosine of mixed angles is deterministic."""
        import math

        query_vec = [1.0, 0.0]
        # 45-degree angle → cosine = cos(45°) = 1/sqrt(2)
        v45 = [1.0, 1.0]  # not normalised — cosine formula handles it
        doc_vecs = [v45, v45, v45]

        gen = FakeGenerator([{"questions": ["Q1?", "Q2?", "Q3?"]}])
        emb = FakeEmbedder(query_vec=query_vec, doc_vecs=doc_vecs)

        score = answer_relevancy("Q?", "A.", gen, emb)
        expected = 1.0 / math.sqrt(2)
        assert score == pytest.approx(expected, abs=1e-6)

    def test_no_questions_generated(self) -> None:
        gen = FakeGenerator([{"questions": []}])
        emb = FakeEmbedder(query_vec=[1.0, 0.0], doc_vecs=[])
        score = answer_relevancy("Q?", "A.", gen, emb)
        assert score == pytest.approx(0.0)


class TestContextPrecision:
    """Test context_precision with a FakeGenerator."""

    def test_all_relevant(self) -> None:
        """Two contexts both relevant → CP = 1.0."""
        gen = FakeGenerator(
            [
                {"relevant": True},  # context 1
                {"relevant": True},  # context 2
            ]
        )
        score = context_precision("Q", "A", ["ctx1", "ctx2"], gen)
        assert score == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        gen = FakeGenerator([{"relevant": False}, {"relevant": False}])
        score = context_precision("Q", "A", ["ctx1", "ctx2"], gen)
        assert score == pytest.approx(0.0)

    def test_second_relevant(self) -> None:
        """Only rank-2 context is relevant.

        P@1=0 (not counted), P@2=0.5 (counted).
        CP = (0 + 0.5) / 1 = 0.5
        """
        gen = FakeGenerator([{"relevant": False}, {"relevant": True}])
        score = context_precision("Q", "A", ["ctx1", "ctx2"], gen)
        assert score == pytest.approx(0.5)

    def test_empty_contexts(self) -> None:
        gen = FakeGenerator([])
        score = context_precision("Q", "A", [], gen)
        assert score == pytest.approx(0.0)


class TestContextRecall:
    def test_all_attributable(self) -> None:
        gen = FakeGenerator(
            [
                {"statements": ["Stmt A", "Stmt B"]},
                {
                    "verdicts": [
                        {"statement": "Stmt A", "attributable": True},
                        {"statement": "Stmt B", "attributable": True},
                    ]
                },
            ]
        )
        score = context_recall("Q", "GT", ["ctx"], gen)
        assert score == pytest.approx(1.0)

    def test_half_attributable(self) -> None:
        gen = FakeGenerator(
            [
                {"statements": ["Stmt A", "Stmt B"]},
                {
                    "verdicts": [
                        {"statement": "Stmt A", "attributable": True},
                        {"statement": "Stmt B", "attributable": False},
                    ]
                },
            ]
        )
        score = context_recall("Q", "GT", ["ctx"], gen)
        assert score == pytest.approx(0.5)

    def test_no_statements(self) -> None:
        gen = FakeGenerator([{"statements": []}])
        score = context_recall("Q", "GT", ["ctx"], gen)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fast_subset
# ---------------------------------------------------------------------------


class TestFastSubset:
    def test_small_list_unchanged(self) -> None:
        items = list(range(5))
        result = fast_subset(items, n=10, seed=0)
        assert len(result) == 5

    def test_deterministic(self) -> None:
        items = list(range(100))
        r1 = fast_subset(items, n=15, seed=0)
        r2 = fast_subset(items, n=15, seed=0)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        items = list(range(100))
        r1 = fast_subset(items, n=15, seed=0)
        r2 = fast_subset(items, n=15, seed=1)
        assert r1 != r2

    def test_correct_length(self) -> None:
        items = list(range(100))
        result = fast_subset(items, n=15, seed=0)
        assert len(result) == 15
