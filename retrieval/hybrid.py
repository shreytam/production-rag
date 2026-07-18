"""Retrievers.

`DenseRetriever` is the naive baseline (embed -> ACL-scoped vector search).
`HybridRetriever` is the production path: dense + BM25, each ACL-scoped at the
store, fused with RRF, then cross-encoder reranked. ACL is pushed into each
store's search() so candidates are tenant-scoped BEFORE fusion — never dropped
afterward.
"""

from __future__ import annotations

from core.interfaces import Embedder, Reranker, SparseRetriever, VectorStore
from core.rrf import reciprocal_rank_fusion
from core.types import Query, ScoredChunk


class DenseRetriever:
    """Baseline: dense vector search only."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: Query) -> list[ScoredChunk]:
        qvec = self.embedder.embed_query(query.text)
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
        qvec = self.embedder.embed_query(query.text)
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
