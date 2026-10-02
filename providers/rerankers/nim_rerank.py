"""NVIDIA NIM reranking provider."""

from __future__ import annotations

import httpx

from core.types import ScoredChunk
from providers.rerankers._common import retry_transient


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

        from providers.rerankers._common import normalize_candidates

        scored = [(int(item["index"]), _score(item)) for item in rankings]
        return normalize_candidates(chunks, scored, top_n)
