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


# --- defect 3: orphans / manifest-independent delete -------------------------

class _BoomSparse(BM25Retriever):
    def add(self, chunks):
        raise RuntimeError("sparse down")


def test_failed_new_ingest_cleans_up_upserted_points(tmp_path):
    import pytest
    store = InMemoryVectorStore()
    ing = _ing(tmp_path, store, _BoomSparse())
    with pytest.raises(RuntimeError):
        ing.ingest_document("t1", "d1", [_c()], ACLContext(tenant_id="t1"))
    assert store.chunks == []


def test_delete_document_without_manifest_removes_orphans(tmp_path):
    store, sparse = InMemoryVectorStore(), BM25Retriever()
    ing = _ing(tmp_path, store, sparse)
    acl = ACLContext(tenant_id="t1")
    ing.ingest_document("t1", "d1", [_c(), _c("d1::1", "beta", ordinal=1)], acl)
    ing._manifest.delete("t1", "d1")  # simulate lost manifest
    assert ing.delete_document("t1", "d1", acl) == 2
    assert store.chunks == []
    assert sparse.snapshot("t1") == []


def test_delete_by_doc_is_tenant_scoped(tmp_path):
    store = InMemoryVectorStore()
    ing = _ing(tmp_path, store)
    ing.ingest_document("t1", "d1", [_c()], ACLContext(tenant_id="t1"))
    ing.ingest_document("t2", "d1", [_c(tenant="t2")], ACLContext(tenant_id="t2"))
    ing.delete_document("t1", "d1", ACLContext(tenant_id="t1"))
    assert [c.tenant_id for c in store.chunks] == ["t2"]


# --- defect 6a: upsert batching ----------------------------------------------

def test_upserts_are_batched(tmp_path):
    from ingest import incremental
    store = InMemoryVectorStore()
    sizes = []
    orig = store.upsert
    store.upsert = lambda cs: (sizes.append(len(cs)), orig(cs))[1]
    ing = _ing(tmp_path, store)
    n = incremental._UPSERT_BATCH * 2 + 5
    ing.ingest_document("t1", "d1",
                        [_c(f"d1::{i}", f"w{i}", ordinal=i) for i in range(n)],
                        ACLContext(tenant_id="t1"))
    assert max(sizes) <= incremental._UPSERT_BATCH and len(sizes) == 3
    assert len(store.chunks) == n


# --- defect 6b: embedder integrity -------------------------------------------

class _Item:
    def __init__(self, index, emb):
        self.index, self.embedding = index, emb


def _embedder(items_fn):
    from types import SimpleNamespace
    from core.config import Settings
    from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
    e = OpenAICompatibleEmbedder(Settings(embed_base_url="http://x", embed_api_key="k"))
    e._client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(data=items_fn(kw["input"]))))
    return e


def test_embedder_sorts_by_index():
    e = _embedder(lambda inp: [_Item(1, [1.0]), _Item(0, [0.0])])
    assert e.embed_documents(["a", "b"]) == [[0.0], [1.0]]


def test_embedder_raises_on_count_mismatch():
    import pytest
    e = _embedder(lambda inp: [_Item(0, [0.0])])
    with pytest.raises(ValueError):
        e.embed_documents(["a", "b"])


# --- defect 7: chunker --------------------------------------------------------

def test_paragraphs_do_not_fuse():
    from core.types import Document
    from ingest.chunking import chunk_document
    doc = Document(doc_id="d", tenant_id="t", text="end.\n\nNext paragraph")
    text = chunk_document(doc, max_tokens=64, overlap=0)[0].text
    assert "end.Next" not in text and "end." in text and "Next paragraph" in text


def test_no_replacement_chars_from_slicing():
    from core.types import Document
    from ingest.chunking import chunk_document
    doc = Document(doc_id="d", tenant_id="t", text="日本語のテキスト😀" * 200)
    chunks = chunk_document(doc, max_tokens=17, overlap=5)
    assert len(chunks) > 3
    assert all("�" not in c.text for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"d::{i:06d}" for i in range(len(chunks))]


# --- defect 8: contextual cache -----------------------------------------------

def test_contextual_cache_key_depends_on_model_and_separator():
    from ingest.contextual import _cache_key
    assert _cache_key("m1", "ab", "c") != _cache_key("m1", "a", "bc")
    assert _cache_key("m1", "a", "b") != _cache_key("m2", "a", "b")


def test_contextual_cache_write_is_atomic(tmp_path):
    from core.config import Settings
    from ingest.contextual import ContextualPrefixer
    from tests._fakes import RecordingGenerator
    p = ContextualPrefixer(RecordingGenerator(text="ctx"), cache_dir=tmp_path,
                           settings=Settings(pii_mode="keep"))
    assert p.prefix_for("doc", "chunk", doc_id="d") == "ctx"
    files = list(p._cache_dir.iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"
    assert p.prefix_for("doc", "chunk", doc_id="d") == "ctx"
    assert len(p._gen.calls) == 1


# --- defects 4 & 5: worker -----------------------------------------------------

def _deps(tmp_path, registry=None):
    from core.config import Settings
    from ingest.parsers.base import ParserRegistry
    from ingest.worker import IngestDeps
    from providers.docstore.memory import InMemoryDocumentRegistry
    from tests.test_ingest_delete_worker import DictBlobs
    return IngestDeps(
        registry=registry or InMemoryDocumentRegistry(), blobs=DictBlobs(),
        parsers=ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000),
        ingestor=_ing(tmp_path), settings=Settings(pii_mode="keep"))


def _rec(status, doc="d1", tenant="t"):
    from core.types import DocumentRecord
    return DocumentRecord(document_id=doc, tenant_id=tenant, filename="f.txt",
                          content_type="text/plain", size_bytes=5, status=status,
                          blob_key=f"{tenant}/{doc}")


def test_empty_document_is_failed_not_ready(tmp_path):
    from core.types import DocumentStatus
    from ingest.worker import run_ingest
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"   \n\n  ")
    deps.registry.create(_rec(DocumentStatus.PROCESSING))
    run_ingest(deps, "d1")
    r = deps.registry.get_privileged("d1")
    assert r.status == DocumentStatus.FAILED and "no extractable text" in r.error


def test_run_ingest_skips_deleting_document(tmp_path):
    from core.types import DocumentStatus
    from ingest.worker import run_ingest
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta")
    deps.registry.create(_rec(DocumentStatus.DELETING))
    run_ingest(deps, "d1")
    assert deps.ingestor._store.chunks == []
    assert deps.registry.get_privileged("d1").status == DocumentStatus.DELETING


def test_ingest_racing_with_delete_leaves_no_orphans(tmp_path):
    """Delete flips status mid-flight; the finished ingest must purge itself."""
    from core.types import DocumentStatus
    from ingest.worker import run_ingest
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta")
    deps.registry.create(_rec(DocumentStatus.PROCESSING))
    orig = deps.ingestor.ingest_document

    def racing(*a, **k):
        n = orig(*a, **k)
        deps.registry.set_status("d1", "t", DocumentStatus.DELETING)
        return n
    deps.ingestor.ingest_document = racing
    run_ingest(deps, "d1")
    assert deps.ingestor._store.chunks == []
    assert deps.registry.get_privileged("d1").status == DocumentStatus.DELETING


def test_list_stale_memory_registry():
    from datetime import datetime, timedelta, timezone
    from core.types import DocumentStatus
    from providers.docstore.memory import InMemoryDocumentRegistry
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    reg = InMemoryDocumentRegistry(clock=lambda: now[0])
    reg.create(_rec(DocumentStatus.PROCESSING, "old"))
    reg.create(_rec(DocumentStatus.READY, "ready"))
    now[0] += timedelta(hours=2)
    reg.create(_rec(DocumentStatus.PROCESSING, "new"))
    stale = reg.list_stale((DocumentStatus.PROCESSING, DocumentStatus.DELETING),
                           timedelta(hours=1))
    assert [r.document_id for r in stale] == ["old"]


def test_sweep_stale_marks_failed(tmp_path):
    from datetime import datetime, timedelta, timezone
    from core.types import DocumentStatus
    from ingest.worker import sweep_stale_documents
    from providers.docstore.memory import InMemoryDocumentRegistry
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    reg = InMemoryDocumentRegistry(clock=lambda: now[0])
    deps = _deps(tmp_path, reg)
    reg.create(_rec(DocumentStatus.PROCESSING, "a"))
    reg.create(_rec(DocumentStatus.DELETING, "b"))
    now[0] += timedelta(hours=5)
    assert sweep_stale_documents(deps) == 2
    for d in ("a", "b"):
        r = reg.get_privileged(d)
        assert r.status == DocumentStatus.FAILED and r.error
    assert deps.settings.ingest_stale_after_seconds > deps.settings.ingest_job_timeout_seconds


def test_worker_settings_and_async_wrappers():
    import asyncio
    import threading
    from ingest import worker
    ws = worker.WorkerSettings
    assert ws.job_timeout > 0 and ws.max_tries >= 1
    assert ws.cron_jobs

    seen = {}
    orig = worker.run_ingest
    worker.run_ingest = lambda deps, doc: seen.setdefault("t", threading.current_thread())
    try:
        asyncio.run(worker.ingest_document({"deps": object()}, "d1"))
    finally:
        worker.run_ingest = orig
    assert seen["t"] is not threading.main_thread()
