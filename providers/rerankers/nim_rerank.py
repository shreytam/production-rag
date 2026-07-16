"""NVIDIA NIM reranking provider."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.types import RetrievalSource, ScoredChunk


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


class NIMReranker:
    """Reranker backed by the NVIDIA NIM ranking endpoint.

    Args:
        model: NIM model identifier (e.g. "nvidia/nv-rerankqa-mistral-4b-v3").
        base_url: Base URL of the NIM service (e.g. "https://ai.api.nvidia.com/v1/retrieval").
        api_key: Bearer token for authentication.
    """

    _TIMEOUT = 10.0  # seconds

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        payload = {
            "model": self._model,
            "query": {"text": query},
            "passages": [{"text": c.chunk.text} for c in chunks],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self._TIMEOUT) as client:
            response = client.post(
                f"{self._base_url}/ranking",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()

        rankings: list[dict] = data.get("rankings", [])

        def _score(item: dict) -> float:
            # NIM may return "logit" or "score"
            for key in ("logit", "score", "relevance_score"):
                if key in item:
                    return float(item[key])
            return 0.0

        # Build raw scores list matching input chunks order
        raw_scores_map = {}
        for item in rankings:
            idx = int(item["index"])
            raw_scores_map[idx] = _score(item)

        raw_scores = []
        for idx in range(len(chunks)):
            raw_scores.append(raw_scores_map.get(idx, 0.0))

        from providers.rerankers._common import min_max_normalize
        normalized = min_max_normalize(raw_scores)

        scored = sorted(
            zip(normalized, chunks),
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
