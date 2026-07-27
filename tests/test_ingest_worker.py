import asyncio

from core.config import Settings
from core.types import DocumentRecord, DocumentStatus
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, on_startup, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore, StrictInMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k):
        if k not in self.d:
            raise KeyError(k)
        return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def _deps(tmp_path, store=None):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    ingestor = IncrementalIngestor(FakeEmbedder(), store or InMemoryVectorStore(),
                                   BM25Retriever(), JsonlManifestStore(str(tmp_path)))
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    settings = Settings(pii_mode="keep")
    return IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=settings)


def _seed(deps, did="doc1", tenant="t1", body=b"alpha beta gamma"):
    deps.blobs.put(f"{tenant}/{did}", body)
    deps.registry.create(DocumentRecord(
        document_id=did, tenant_id=tenant, filename="f.txt", content_type="text/plain",
        size_bytes=len(body), status=DocumentStatus.PROCESSING, blob_key=f"{tenant}/{did}"))


def test_worker_moves_document_to_ready(tmp_path):
    deps = _deps(tmp_path)
    _seed(deps)
    run_ingest(deps, "doc1")
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.READY
    assert rec.chunk_count >= 1
    assert len(deps.ingestor._store.chunks) >= 1


def test_worker_marks_failed_on_parse_error(tmp_path):
    deps = _deps(tmp_path)
    _seed(deps, body=b"x")
    # Force a parser failure: unknown type slips past by mutating the record.
    deps.registry._rows["doc1"] = deps.registry._rows["doc1"].model_copy(
        update={"content_type": "application/x-nope"})
    run_ingest(deps, "doc1")
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.FAILED
    assert rec.error
    assert len(deps.ingestor._store.chunks) == 0  # no partial index


def test_worker_fails_closed_when_startup_never_ensured_collection(tmp_path):
    """Regression guard: without the worker startup hook, the first upload
    upserts into a collection that was never created and the document lands
    in `failed` — exactly the P0 bug. Proves StrictInMemoryVectorStore (the
    faithful fake) actually catches it, unlike the lenient InMemoryVectorStore
    stub which accepts upsert() unconditionally."""
    deps = _deps(tmp_path, store=StrictInMemoryVectorStore())
    _seed(deps)
    run_ingest(deps, "doc1")  # no on_startup call: collection was never ensured
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.FAILED
    assert rec.error


def test_worker_startup_ensures_collection_then_first_upload_reaches_ready(tmp_path):
    """The real worker/API path: on_startup() runs once (as arq calls it before
    polling jobs), then the first document ingest reaches `ready` because the
    collection now exists."""
    deps = _deps(tmp_path, store=StrictInMemoryVectorStore())
    _seed(deps)

    asyncio.run(on_startup({"deps": deps}))
    assert deps.ingestor._store.collection_ensured is True

    run_ingest(deps, "doc1")
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.READY
    assert rec.chunk_count >= 1


# ---------------------------------------------------------------------------
# Live Qdrant: the API/worker path against a REAL, empty collection. Skipped
# (or failed, under RAG_REQUIRE_LIVE_STORES) when Qdrant is unreachable — same
# convention as tests/test_stores_acl.py and tests/test_multitenant_isolation.py.
# ---------------------------------------------------------------------------

def _qdrant_reachable(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=url, timeout=2).get_collections()
        return True
    except Exception:  # noqa: BLE001 — only used to decide skip vs run
        return False


def test_qdrant_live_first_upload_reaches_ready(require_live_or_fail, tmp_path):
    """Drives the real QdrantVectorStore, starting from a collection that does
    not exist yet — reproducing the exact production scenario the P0 bug hit:
    point a fresh Qdrant at the API and the first upload must still reach
    `ready`, because the worker startup hook creates the collection first."""
    from providers.vectorstores.qdrant_store import QdrantVectorStore

    settings = Settings(qdrant_collection="test_p0_ingest_startup")
    require_live_or_fail(_qdrant_reachable(settings.qdrant_url), "Qdrant")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    try:
        client.delete_collection(settings.qdrant_collection)  # empty slate
    except Exception:
        pass

    store = QdrantVectorStore(settings)
    ingestor = IncrementalIngestor(FakeEmbedder(), store, BM25Retriever(),
                                   JsonlManifestStore(str(tmp_path)))
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    deps = IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=Settings(pii_mode="keep"))
    _seed(deps)

    asyncio.run(on_startup({"deps": deps}))  # the worker startup hook
    run_ingest(deps, "doc1")

    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.READY
    assert rec.chunk_count >= 1
