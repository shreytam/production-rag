"""Retrievers.

`DenseRetriever` is the naive baseline (embed -> ACL-scoped vector search).
`HybridRetriever` is the production path: dense + BM25, each ACL-scoped at the
store, fused with RRF, then cross-encoder reranked. ACL is pushed into each
store's search() so candidates are tenant-scoped BEFORE fusion — never dropped
afterward.
"""

from __future__ import annotations

from core.config import get_settings
from core.interfaces import Embedder, Reranker, SparseRetriever, VectorStore
from core.rrf import reciprocal_rank_fusion
from core.types import Query, ScoredChunk


_TRACER = None


def _embed_traced(embedder: Embedder, query: Query):
    """Embed the query inside an `embedding`-typed Langfuse observation.

    Always traced (independent of the semantic-cache flag) so every query shows
    its embedding call: model + dim. Best-effort — tracing must never break
    retrieval, and no-op tracers make this free when Langfuse is disabled.
    """
    from observability.langfuse_tracing import Tracer

    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer(get_settings())
    with _TRACER.span(
        "dense.embed_query", as_type="embedding",
        model=get_settings().embed_model,
    ) as s_emb:
        qvec = embedder.embed_query(query.text)
        if qvec is not None:
            s_emb.update(dim=len(qvec))
    return qvec


class DenseRetriever:
    """Baseline: dense vector search only."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: Query) -> list[ScoredChunk]:
        qvec = _embed_traced(self.embedder, query)
        return self.vector_store.search(qvec, query.top_k, query.acl,
                                        collection_id=query.collection_id)


class HybridRetriever:
    """Dense + sparse -> RRF -> rerank, all ACL-scoped at the source."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse: SparseRetriever,
        reranker: Reranker,
        rrf_k: int = 60,
        fuse_window: int = 40,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse = sparse
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.fuse_window = fuse_window

    def retrieve(self, query: Query) -> list[ScoredChunk]:
        qvec = _embed_traced(self.embedder, query)
        dense = self.vector_store.search(
            qvec, query.top_k, query.acl, collection_id=query.collection_id
        )
        sparse = self.sparse.search(
            query.text, query.top_k, query.acl, collection_id=query.collection_id
        )
        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k)
        if not fused:
            return []
        window = fused[: self.fuse_window]
        return self.reranker.rerank(query.text, window, query.rerank_top_n)
