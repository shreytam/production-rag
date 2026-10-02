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


def _chunk_uuid(tenant_id: str, chunk_id: str) -> str:
    """Point id is tenant-scoped so two tenants can never overwrite each
    other's points, even if their chunk_ids collide."""
    return str(uuid.uuid5(_NS, f"{tenant_id}\x00{chunk_id}"))


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
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
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

        # Payload index on doc_id for delete-by-document
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="doc_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
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
                    id=_chunk_uuid(chunk.tenant_id, chunk.chunk_id),
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

    @staticmethod
    def _tenant_cond(tenant_id: str) -> qm.FieldCondition:
        return qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        """Delete points, scoped to the caller's tenant.

        Deliberately tenant-scoped, NOT tag-scoped: ingest is a trusted write
        path whose ACL carries no tags, and a tag filter would silently skip
        restricted points (leaving them live)."""
        if not chunk_ids:
            return
        ids = [_chunk_uuid(acl.tenant_id, cid) for cid in chunk_ids]
        combined = qm.Filter(must=[qm.HasIdCondition(has_id=ids),
                                   self._tenant_cond(acl.tenant_id)])
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=combined),
        )

    def delete_by_doc(self, tenant_id: str, doc_id: str) -> list[str]:
        """Delete every point of (tenant_id, doc_id) by payload filter, without
        consulting any manifest. Returns the chunk_ids that were present."""
        flt = qm.Filter(must=[
            self._tenant_cond(tenant_id),
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ])
        chunk_ids: list[str] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection, scroll_filter=flt,
                limit=256, offset=offset, with_payload=["chunk_id"],
                with_vectors=False,
            )
            chunk_ids.extend(
                p.payload["chunk_id"] for p in points
                if p.payload and "chunk_id" in p.payload)
            if offset is None:
                break
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=flt),
        )
        return chunk_ids

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        """Patch point payloads, scoped to the caller's tenant (see delete())."""
        for chunk_id, payload in updates.items():
            pt = _chunk_uuid(acl.tenant_id, chunk_id)
            combined = qm.Filter(must=[qm.HasIdCondition(has_id=[pt]),
                                       self._tenant_cond(acl.tenant_id)])
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
