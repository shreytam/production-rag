"""BM25 sparse retriever implementation.

ACL enforcement strategy:
  - At index time, chunks are grouped by tenant_id into SEPARATE BM25 indices.
  - At search time, only the index for acl.tenant_id is consulted — cross-tenant
    chunks are never scored (not even ranked and discarded).
  - acl_predicate() is applied after scoring to enforce tag-level visibility
    within the tenant.

This two-level strategy means cross-tenant isolation is structural (wrong index),
and tag-scoping is applied on the already-tenant-isolated candidate set.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from core.types import ACLContext, Chunk, RetrievalSource, ScoredChunk
from retrieval.acl import acl_predicate


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Retriever:
    """In-memory BM25 retriever with per-tenant index isolation.

    Index chunks with .index(); query with .search() providing an ACLContext.
    """

    def __init__(self) -> None:
        # tenant_id -> (BM25Okapi instance, list[Chunk])
        self._indices: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}

    def index(self, chunks: list[Chunk]) -> None:
        """Build per-tenant BM25 indices from the provided chunks.

        Can be called multiple times; subsequent calls replace the index
        (full re-index semantics — suitable for batch ingest).
        """
        # Group by tenant
        tenant_chunks: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            tenant_chunks.setdefault(chunk.tenant_id, []).append(chunk)

        # Build one BM25Okapi per tenant
        for tenant_id, t_chunks in tenant_chunks.items():
            corpus = [_tokenize(c.embed_text) for c in t_chunks]
            self._indices[tenant_id] = (BM25Okapi(corpus), t_chunks)

    def search(self, query: str, top_k: int, acl: ACLContext) -> list[ScoredChunk]:
        """Return top-k chunks for the caller's tenant, filtered by ACL tags.

        Cross-tenant isolation: we only access the acl.tenant_id index.
        Tag scoping: acl_predicate filters out tag-restricted chunks the
        caller cannot see.
        """
        if acl.tenant_id not in self._indices:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        bm25, chunks = self._indices[acl.tenant_id]
        scores = bm25.get_scores(tokens)

        # Build (score, chunk) pairs, apply ACL predicate, sort descending
        predicate = acl_predicate(acl)
        candidates: list[tuple[float, Chunk]] = [
            (float(scores[i]), chunk)
            for i, chunk in enumerate(chunks)
            if predicate(chunk)
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[ScoredChunk] = []
        for rank, (score, chunk) in enumerate(candidates[:top_k], start=1):
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    source=RetrievalSource.SPARSE,
                    rank=rank,
                )
            )
        return results
