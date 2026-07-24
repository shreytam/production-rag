from core.types import DocumentRecord, DocumentStatus
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_delete, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from core.config import Settings
from tests._fakes import FakeEmbedder, InMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k):
        if k not in self.d:
            raise KeyError(k)
        return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def _deps(tmp_path):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    ingestor = IncrementalIngestor(FakeEmbedder(), InMemoryVectorStore(),
                                   BM25Retriever(), JsonlManifestStore(str(tmp_path)))
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    return IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=Settings(pii_mode="keep"))


def test_run_ingest_stamps_collection_id(tmp_path):
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta gamma")
    deps.registry.create(DocumentRecord(document_id="d1", tenant_id="t", filename="f.txt",
        content_type="text/plain", size_bytes=15, status=DocumentStatus.PROCESSING,
        blob_key="t/d1", collection_id="A"))
    run_ingest(deps, "d1")
    assert deps.ingestor._store.chunks
    assert all(c.collection_id == "A" for c in deps.ingestor._store.chunks)


def test_run_delete_purges_everything(tmp_path):
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta gamma")
    deps.registry.create(DocumentRecord(document_id="d1", tenant_id="t", filename="f.txt",
        content_type="text/plain", size_bytes=15, status=DocumentStatus.PROCESSING,
        blob_key="t/d1", collection_id="A"))
    run_ingest(deps, "d1")
    run_delete(deps, "d1")
    assert deps.registry.get_privileged("d1") is None      # row gone (last)
    assert "t/d1" not in deps.blobs.d                        # blob gone
    assert deps.ingestor._store.chunks == []                # chunks gone
