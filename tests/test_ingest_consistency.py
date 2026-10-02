"""Regression tests for ingestion consistency defects (ACL/meta, tenant ids,
orphans, worker robustness, empty docs, batching, chunker, contextual cache)."""
from __future__ import annotations

from pathlib import Path

from core.types import ACLContext, Chunk
from ingest.incremental import IncrementalIngestor
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _c(cid="d1::0", text="alpha", tenant="t1", tags=(), coll="", doc="d1", ordinal=0):
    return Chunk(chunk_id=cid, doc_id=doc, text=text, tenant_id=tenant,
                 acl_tags=tags, collection_id=coll, ordinal=ordinal)


def _ing(tmp_path, store=None, sparse=None):
    return IncrementalIngestor(FakeEmbedder(), store or InMemoryVectorStore(),
                               sparse or BM25Retriever(),
                               JsonlManifestStore(str(tmp_path)))


# --- defect 2: tenant-scoped ids -------------------------------------------

def test_cli_doc_id_includes_tenant():
    from ingest.run import _doc_id_for
    p = Path("/tmp/x.pdf")
    assert _doc_id_for(p, "a") != _doc_id_for(p, "b")
    assert _doc_id_for(p, "a") == _doc_id_for(p, "a")


def test_qdrant_point_id_includes_tenant():
    from providers.vectorstores.qdrant_store import _chunk_uuid
    assert _chunk_uuid("a", "d::0") != _chunk_uuid("b", "d::0")
    assert _chunk_uuid("a", "d::0") == _chunk_uuid("a", "d::0")


# --- defect 1: ACL / collection change on re-ingest -------------------------

def test_reingest_with_new_acl_tags_and_collection_updates_everything(tmp_path):
    store, sparse = InMemoryVectorStore(), BM25Retriever()
    ing = _ing(tmp_path, store, sparse)
    ing.ingest_document("t1", "d1", [_c()], ACLContext(tenant_id="t1"))
    assert store.chunks[0].acl_tags == ()
    ing.ingest_document("t1", "d1", [_c(tags=("finance",), coll="C")],
                        ACLContext(tenant_id="t1", acl_tags=("finance",)))
    assert len(store.chunks) == 1
    assert store.chunks[0].acl_tags == ("finance",)
    assert store.chunks[0].collection_id == "C"
    sp = sparse.snapshot("t1")
    assert len(sp) == 1 and sp[0].acl_tags == ("finance",) and sp[0].collection_id == "C"
    # an anonymous caller must no longer see it
    assert sparse.search("alpha", 5, ACLContext(tenant_id="t1")) == []
