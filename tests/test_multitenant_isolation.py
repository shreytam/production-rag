"""Multi-tenant isolation: tenant A must never retrieve tenant B's documents.

Proven at three layers — dense store, BM25, and the full hybrid pipeline — plus
live Qdrant/pgvector when reachable. The decisive trick: tenant-B chunks are
written with text that strongly matches the query, so any leak is a real ACL
failure, not an accident of ranking.
"""

from __future__ import annotations

import pytest

from core.config import get_settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Chunk
from generation.grounded_generator import GeneratedAnswer, GroundedGenerator
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import DIM, FakeEmbedder, FakeReranker, InMemoryVectorStore, RecordingGenerator

QUERY = "quarterly revenue secret figures"


def _corpus() -> list[Chunk]:
    return [
        Chunk(chunk_id="pub::0", doc_id="pub", text="quarterly revenue summary public", tenant_id="public"),
        Chunk(chunk_id="a::0", doc_id="a", text="quarterly revenue secret figures tenant a", tenant_id="tenant_a"),
        # Tenant-B chunk engineered to match the query BETTER than anyone else:
        Chunk(chunk_id="b::0", doc_id="b", text="quarterly revenue secret figures tenant b confidential", tenant_id="tenant_b"),
        # Tag-scoped chunk inside tenant_a:
        Chunk(chunk_id="a::sec", doc_id="asec", text="quarterly revenue secret board only", tenant_id="tenant_a", acl_tags=("board",)),
    ]


def _embedded():
    emb = FakeEmbedder()
    chunks = _corpus()
    vecs = emb.embed_documents([c.embed_text for c in chunks])
    return emb, [c.model_copy(update={"embedding": v}) for c, v in zip(chunks, vecs)]


def test_dense_store_excludes_other_tenant():
    emb, chunks = _embedded()
    store = InMemoryVectorStore()
    store.upsert(chunks)
    hits = store.search(emb.embed_query(QUERY), top_k=10, acl=ACLContext(tenant_id="tenant_a"))
    tenants = {h.chunk.tenant_id for h in hits}
    assert "tenant_b" not in tenants
    assert tenants <= {"tenant_a"}  # public excluded too: different tenant


def test_bm25_excludes_other_tenant_even_when_b_matches_best():
    bm25 = BM25Retriever()
    bm25.index(_corpus())
    hits = bm25.search(QUERY, top_k=10, acl=ACLContext(tenant_id="tenant_a"))
    assert hits, "tenant_a should retrieve its own matching chunk"
    assert all(h.chunk.tenant_id == "tenant_a" for h in hits)
    assert "b::0" not in {h.chunk_id for h in hits}


def test_tag_scoping_within_tenant():
    emb, chunks = _embedded()
    store = InMemoryVectorStore()
    store.upsert(chunks)
    # Caller in tenant_a with no tags cannot see the board-only chunk.
    untagged = store.search(emb.embed_query(QUERY), 10, ACLContext(tenant_id="tenant_a"))
    assert "a::sec" not in {h.chunk_id for h in untagged}
    # Caller holding the 'board' tag can.
    tagged = store.search(emb.embed_query(QUERY), 10, ACLContext(tenant_id="tenant_a", acl_tags=("board",)))
    assert "a::sec" in {h.chunk_id for h in tagged}


def test_full_pipeline_no_cross_tenant_leak():
    emb, chunks = _embedded()
    store = InMemoryVectorStore()
    store.upsert(chunks)
    bm25 = BM25Retriever()
    bm25.index(chunks)
    retriever = HybridRetriever(emb, store, bm25, FakeReranker(), rrf_k=60)
    grounded = GroundedGenerator(
        RecordingGenerator(parsed=GeneratedAnswer(answer="see [1]", citations=[1]).model_dump())
    )
    pipe = RAGPipeline(retriever, grounded, get_settings())

    out = pipe.run(QUERY, acl=ACLContext(tenant_id="tenant_a"))
    assert "b" not in out["retrieved_ids"], "tenant_b doc leaked into tenant_a results"
    assert all(sc.chunk.tenant_id == "tenant_a" for sc in out["answer_obj"].contexts)


# --------------------------------------------------------------------------
# Live stores (skip when the backend is not running)
# --------------------------------------------------------------------------


def _live_isolation_check(store):
    """Exercise the REAL store: build collection, upsert, ACL-scoped search.

    Connectivity is checked separately (skip if down); once we know the server is
    up, any failure here is a real bug in the store/filter code, NOT a skip.
    """
    store.ensure_collection(DIM)
    _, chunks = _embedded()
    store.upsert(chunks)
    hits = store.search(FakeEmbedder().embed_query(QUERY), top_k=10, acl=ACLContext(tenant_id="tenant_a"))
    assert all(h.chunk.tenant_id == "tenant_a" for h in hits)
    assert "b::0" not in {h.chunk_id for h in hits}
    # The board-tagged chunk is hidden from an untagged tenant_a caller.
    assert "a::sec" not in {h.chunk_id for h in hits}


def _qdrant_reachable(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=url, timeout=2).get_collections()
        return True
    except Exception:  # noqa: BLE001 — only used to decide skip vs run
        return False


def _pg_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


def test_qdrant_live_isolation(require_live_or_fail):
    from providers.vectorstores.qdrant_store import QdrantVectorStore

    s = get_settings().model_copy(update={"qdrant_collection": "rag_isolation_test"})
    require_live_or_fail(_qdrant_reachable(s.qdrant_url), "Qdrant")
    _live_isolation_check(QdrantVectorStore(s))  # bugs here FAIL, not skip


def test_pgvector_live_isolation(require_live_or_fail):
    from providers.vectorstores.pgvector_store import PgVectorStore

    s = get_settings().model_copy(update={"pg_table": "rag_isolation_test"})
    require_live_or_fail(_pg_reachable(s.pg_dsn), "Postgres")
    _live_isolation_check(PgVectorStore(s))  # bugs here FAIL, not skip
