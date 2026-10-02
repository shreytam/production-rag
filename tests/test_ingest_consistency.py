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


