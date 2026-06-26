"""Offline tests for reranker providers.

No model downloads. No network calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.types import Chunk, RetrievalSource, ScoredChunk
from providers.rerankers.local_cross_encoder import LocalCrossEncoderReranker
from providers.rerankers.nim_rerank import NIMReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        tenant_id="t1",
    )


def _make_scored(chunk_id: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=_make_chunk(chunk_id, text),
        score=score,
        source=RetrievalSource.DENSE,
    )


# ---------------------------------------------------------------------------
# LocalCrossEncoderReranker tests
# ---------------------------------------------------------------------------

class TestLocalCrossEncoderReranker:
    def _chunks(self) -> list[ScoredChunk]:
        """Return 3 chunks whose dense scores are [0.9, 0.5, 0.1] (order A, B, C)."""
        return [
            _make_scored("A", "text about apples", 0.9),
            _make_scored("B", "text about bananas", 0.5),
            _make_scored("C", "text about cherries", 0.1),
        ]

    def test_rerank_reverses_order_via_scorer(self) -> None:
        """Fake scorer assigns scores [0.1, 0.5, 0.9] => order should become C, B, A."""

        def fake_scorer(pairs: list[tuple[str, str]]) -> list[float]:
            # Reverse order: last pair gets highest score
            n = len(pairs)
            return [float(i) / (n - 1) for i in range(n)]  # [0.0, 0.5, 1.0]

        reranker = LocalCrossEncoderReranker(
            "BAAI/bge-reranker-v2-m3", scorer=fake_scorer
        )
        chunks = self._chunks()
        results = reranker.rerank("query", chunks, top_n=2)

        assert len(results) == 2
        # Highest scorer fake score goes to last pair (chunk C) → rank 1
        assert results[0].chunk.chunk_id == "C"
        assert results[1].chunk.chunk_id == "B"
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].source == RetrievalSource.RERANK
        assert results[1].source == RetrievalSource.RERANK
        # Scores should be in descending order
        assert results[0].score > results[1].score

    def test_real_model_never_loaded_when_scorer_provided(self) -> None:
        """CrossEncoder must not be imported/instantiated when scorer is injected."""
        imported_cross_encoder = []

        def fake_scorer(pairs):
            return [1.0] * len(pairs)

        reranker = LocalCrossEncoderReranker(
            "BAAI/bge-reranker-v2-m3", scorer=fake_scorer
        )
        # _model must remain None after rerank when scorer is provided
        reranker.rerank("q", self._chunks(), top_n=2)
        assert reranker._model is None

    def test_top_n_limits_results(self) -> None:
        def fake_scorer(pairs):
            return list(range(len(pairs), 0, -1))  # descending ints

        reranker = LocalCrossEncoderReranker("any", scorer=fake_scorer)
        results = reranker.rerank("q", self._chunks(), top_n=1)
        assert len(results) == 1
        assert results[0].rank == 1

    def test_empty_chunks_returns_empty(self) -> None:
        reranker = LocalCrossEncoderReranker("any", scorer=lambda p: [])
        assert reranker.rerank("q", [], top_n=5) == []

    def test_ranks_are_1_based_sequential(self) -> None:
        scores = [3.0, 1.0, 2.0]

        def fake_scorer(pairs):
            return scores

        reranker = LocalCrossEncoderReranker("any", scorer=fake_scorer)
        results = reranker.rerank("q", self._chunks(), top_n=3)
        assert [r.rank for r in results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# NIMReranker tests
# ---------------------------------------------------------------------------

class TestNIMReranker:
    _BASE_URL = "https://nim.example.com/v1/retrieval"
    _API_KEY = "test-key"
    _MODEL = "nvidia/test-reranker"

    def _reranker(self) -> NIMReranker:
        return NIMReranker(
            model=self._MODEL,
            base_url=self._BASE_URL,
            api_key=self._API_KEY,
        )

    def _chunks(self) -> list[ScoredChunk]:
        return [
            _make_scored("X", "passage about X", 0.3),
            _make_scored("Y", "passage about Y", 0.6),
            _make_scored("Z", "passage about Z", 0.1),
        ]

    def _canned_response(self, rankings: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rankings": rankings}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_nim_ordering_and_top_n(self) -> None:
        """NIM returns rankings that reorder chunks; top_n=2 trims the result."""
        # Simulate NIM ranking: Z (idx=2) best, X (idx=0) second, Y (idx=1) third
        rankings = [
            {"index": 2, "logit": 0.95},
            {"index": 0, "logit": 0.70},
            {"index": 1, "logit": 0.30},
        ]

        with patch("httpx.Client.post", return_value=self._canned_response(rankings)):
            results = self._reranker().rerank("query", self._chunks(), top_n=2)

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "Z"
        assert results[1].chunk.chunk_id == "X"
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].source == RetrievalSource.RERANK
        assert results[0].score == pytest.approx(0.95)

    def test_nim_score_field_fallback(self) -> None:
        """NIM response using 'score' instead of 'logit' should still work."""
        rankings = [
            {"index": 1, "score": 0.88},
            {"index": 0, "score": 0.40},
        ]

        with patch("httpx.Client.post", return_value=self._canned_response(rankings)):
            results = self._reranker().rerank("q", self._chunks()[:2], top_n=2)

        assert results[0].chunk.chunk_id == "Y"
        assert results[1].chunk.chunk_id == "X"

    def test_nim_empty_chunks(self) -> None:
        reranker = self._reranker()
        # No HTTP call expected for empty input
        assert reranker.rerank("q", [], top_n=5) == []

    def test_nim_auth_header_sent(self) -> None:
        """Verify the Authorization header is set correctly."""
        rankings = [{"index": 0, "logit": 1.0}]

        with patch("httpx.Client.post", return_value=self._canned_response(rankings)) as mock_post:
            self._reranker().rerank("q", self._chunks()[:1], top_n=1)

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert headers.get("Authorization") == f"Bearer {self._API_KEY}"

    def test_nim_correct_endpoint(self) -> None:
        """POST must target {base_url}/ranking."""
        rankings = [{"index": 0, "logit": 0.5}]

        with patch("httpx.Client.post", return_value=self._canned_response(rankings)) as mock_post:
            self._reranker().rerank("q", self._chunks()[:1], top_n=1)

        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert url == f"{self._BASE_URL}/ranking"
