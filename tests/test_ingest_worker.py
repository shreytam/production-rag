from core.config import Settings
from core.types import DocumentRecord, DocumentStatus
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k):
        if k not in self.d: raise KeyError(k)
        return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def _deps(tmp_path):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    ingestor = IncrementalIngestor(FakeEmbedder(), InMemoryVectorStore(),
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
