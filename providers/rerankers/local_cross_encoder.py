"""Local cross-encoder reranker using sentence-transformers CrossEncoder."""

from __future__ import annotations

from typing import Callable

from core.types import RetrievalSource, ScoredChunk


class LocalCrossEncoderReranker:
    """Reranker backed by a sentence-transformers CrossEncoder model.

    The model is lazy-loaded on first use so importing this module is cheap
    and safe in offline environments.

    Args:
        model_name: HuggingFace model identifier (e.g. "BAAI/bge-reranker-v2-m3").
        scorer: Optional callable ``(pairs: list[tuple[str, str]]) -> list[float]``.
                When provided it is used instead of loading the real model —
                useful for unit tests that must not touch the network.
    """

    def __init__(
        self,
        model_name: str,
        scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    ) -> None:
        self._model_name = model_name
        self._scorer = scorer
        self._model = None  # lazy-loaded

    def _get_scorer(self) -> Callable[[list[tuple[str, str]]], list[float]]:
        if self._scorer is not None:
            return self._scorer
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(self._model_name)
        return self._model.predict  # type: ignore[return-value]

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        pairs = [(query, c.chunk.text) for c in chunks]
        scorer = self._get_scorer()
        raw_scores: list[float] = list(scorer(pairs))

        scored = sorted(
            zip(raw_scores, chunks),
            key=lambda t: t[0],
            reverse=True,
        )

        results: list[ScoredChunk] = []
        for rank, (score, sc) in enumerate(scored[:top_n], start=1):
            results.append(
                ScoredChunk(
                    chunk=sc.chunk,
                    score=float(score),
                    source=RetrievalSource.RERANK,
                    rank=rank,
                )
            )
        return results
