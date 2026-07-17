from core.types import ACLContext, Chunk
from ingest.incremental import IncrementalIngestor
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _c(cid, ordinal, text, tenant="t1"):
    return Chunk(chunk_id=cid, doc_id="d1", text=text, tenant_id=tenant, ordinal=ordinal)


def _ingestor(tmp_path):
    return IncrementalIngestor(
        FakeEmbedder(), InMemoryVectorStore(), BM25Retriever(),
        JsonlManifestStore(manifest_dir=str(tmp_path)),
    )


def test_first_ingest_embeds_and_indexes(tmp_path):
    ing = _ingestor(tmp_path)
    n = ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], ACLContext(tenant_id="t1"))
    assert n == 1
    assert len(ing._store.chunks) == 1
    assert ing._store.chunks[0].embedding is not None


def test_reingest_unchanged_is_noop(tmp_path):
    ing = _ingestor(tmp_path)
    acl = ACLContext(tenant_id="t1")
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], acl)
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], acl)
    assert len(ing._store.chunks) == 1  # not duplicated


def test_removed_chunk_is_deleted(tmp_path):
    ing = _ingestor(tmp_path)
    acl = ACLContext(tenant_id="t1")
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "a"), _c("d1::1", 1, "b")], acl)
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "a")], acl)  # d1::1 dropped
    ids = {c.chunk_id for c in ing._store.chunks}
    assert ids == {"d1::0"}
