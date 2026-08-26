"""OpenAI-compatible embedder (works with NVIDIA NIM, OpenAI, or any compatible API).

NIM gotcha handled here: NeMo Retriever embedding models (the `nv-embedqa-*` /
E5 family) are asymmetric and REQUIRE an `input_type` of "passage" (indexing) or
"query" (querying). The OpenAI API has no such field, so it is sent via
`extra_body`. We gate this on a NVIDIA base_url so a later swap to real OpenAI
(which would reject the unknown field) keeps working untouched.
"""

from __future__ import annotations

import openai

from core.config import Settings
from core.types import Vector


def _is_nvidia(base_url: str) -> bool:
    return "nvidia.com" in base_url or "integrate.api.nvidia" in base_url


class OpenAICompatibleEmbedder:
    """Embedder that uses the OpenAI embeddings API (or any compatible endpoint)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._needs_input_type = _is_nvidia(settings.embed_base_url)
        self._client = openai.OpenAI(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def dimension(self) -> int:
        return self._settings.embed_dimension

    def _embed(self, texts: list[str], input_type: str) -> list[Vector]:
        """Embed a batch with the correct asymmetric input_type for NIM models."""
        results: list[Vector] = []
        batch_size = self._settings.embed_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs: dict = {
                "model": self._settings.embed_model,
                "input": batch,
                "encoding_format": "float",
            }
            if self._needs_input_type:
                # truncate=END avoids hard failures when a chunk exceeds the
                # model's max sequence length.
                kwargs["extra_body"] = {"input_type": input_type, "truncate": "END"}
            response = self._client.embeddings.create(**kwargs)
            results.extend(item.embedding for item in response.data)

        return results

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed passages for indexing (input_type='passage')."""
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string (input_type='query')."""
        return self._embed([text], input_type="query")[0]
