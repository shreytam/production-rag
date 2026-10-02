"""OpenRouter reranking provider."""

from __future__ import annotations

import httpx

from core.types import ScoredChunk
from providers.rerankers._common import retry_transient


class OpenRouterReranker:
    """Reranker backed by the OpenRouter /rerank endpoint.

    Args:
        model: OpenRouter rerank model (e.g. "nvidia/llama-nemotron-rerank-vl-1b-v2:free").
        base_url: Base URL (defaults to "https://openrouter.ai/api/v1").
        api_key: OpenRouter API key.
    """

    _TIMEOUT = 20.0  # seconds

    def __init__(self, model: str, base_url: str = "https://openrouter.ai/api/v1", api_key: str = "") -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @retry_transient
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
            "query": query,
            "documents": [c.chunk.text for c in chunks],
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self._TIMEOUT) as client:
            endpoint = f"{self._base_url}/rerank" if not self._base_url.endswith("/rerank") else self._base_url
            response = client.post(
                endpoint,
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()

        results: list[dict] = data.get("results", [])

        def _score(item: dict) -> float:
            for key in ("relevance_score", "score", "logit"):
                if key in item:
                    return float(item[key])
            return 0.0

        from providers.rerankers._common import normalize_candidates

        scored = [(int(item["index"]), _score(item)) for item in results]
        return normalize_candidates(chunks, scored, top_n)
