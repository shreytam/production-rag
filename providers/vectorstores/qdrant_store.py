"""Qdrant vector store implementation.

ACL enforcement: every search call passes `query_filter=qdrant_filter(acl)`
so Qdrant applies the tenant + tag filter *before* computing similarity —
no post-filter leakage is possible.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from core.config import Settings
from core.types import ACLContext, Chunk, RetrievalSource, ScoredChunk, Vector
from retrieval.acl import qdrant_filter

# Stable namespace for chunk_id → UUID5 mapping
_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


def _chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def _payload_from_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "tenant_id": chunk.tenant_id,
        "collection_id": chunk.collection_id,
        "acl_tags": list(chunk.acl_tags),
        "acl_open": not bool(chunk.acl_tags),  # True when chunk has no tags
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "title": chunk.title,
        "source": chunk.source,
        "contextual_prefix": chunk.contextual_prefix,
        "metadata": chunk.metadata,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        tenant_id=payload["tenant_id"],
        collection_id=payload.get("collection_id", ""),
        acl_tags=tuple(payload.get("acl_tags") or []),
        text=payload["text"],
        ordinal=payload.get("ordinal", 0),
        title=payload.get("title"),
        source=payload.get("source"),
        contextual_prefix=payload.get("contextual_prefix"),
        metadata=payload.get("metadata") or {},
    )


class QdrantVectorStore:
    """Dense vector store backed by Qdrant.

    ACL is applied as a pre-similarity payload filter on every search().
    The upsert stores `acl_open` (bool) and `acl_tags` (list) in the
    point payload so the filter can be evaluated server-side.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection and tenant_id index if they don't exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=dimension,
                    distance=qm.Distance.COSINE,
                ),
            )

        # Payload index on tenant_id for fast filtering
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="tenant_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Already exists — tolerate
            pass

        # Payload index on collection_id for fast filtering
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="collection_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Already exists — tolerate
            pass

    def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert chunks with their embeddings and ACL payload."""
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            points.append(
                qm.PointStruct(
                    id=_chunk_uuid(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload=_payload_from_chunk(chunk),
                )
            )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        embedding: Vector,
        top_k: int,
        acl: ACLContext,
        *,
        collection_id: str | None = None,
    ) -> list[ScoredChunk]:
        """Search with ACL (and optional collection scoping) applied as a pre-similarity filter."""
        response = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            query_filter=qdrant_filter(acl, collection_id=collection_id),
            limit=top_k,
            with_payload=True,
        )
        scored: list[ScoredChunk] = []
        for rank, hit in enumerate(response.points, start=1):
            chunk = _chunk_from_payload(hit.payload or {})
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(hit.score),
                    source=RetrievalSource.DENSE,
                    rank=rank,
                )
            )
        return scored

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        """Delete points, scoped to the caller's ACL (tenant + tags)."""
        if not chunk_ids:
            return
        ids = [_chunk_uuid(cid) for cid in chunk_ids]
        combined = qm.Filter(must=[qm.HasIdCondition(has_id=ids), qdrant_filter(acl)])
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=combined),
        )

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        """Patch point payloads, scoped to the caller's ACL (tenant + tags)."""
        for chunk_id, payload in updates.items():
            pt = _chunk_uuid(chunk_id)
            combined = qm.Filter(must=[qm.HasIdCondition(has_id=[pt]), qdrant_filter(acl)])
            self._client.set_payload(
                collection_name=self._collection,
                payload=payload,
                points=qm.FilterSelector(filter=combined),
            )

    def count(self, acl: ACLContext | None = None) -> int:
        """Count points, optionally scoped to an ACL context."""
        count_filter = qdrant_filter(acl) if acl is not None else None
        result = self._client.count(
            collection_name=self._collection,
            count_filter=count_filter,
            exact=True,
        )
        return result.count
