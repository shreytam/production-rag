"""Task 2 (P0 ingest fixes): prove that Settings.chunk_max_tokens/chunk_overlap
actually change the chunks the worker (API ingest) path produces end-to-end.

Deliberately does NOT reuse tests/test_ingest_worker.py's `_deps`/`_seed`
helpers (Task 1 territory, out of scope to modify) — mirrors that fixture
pattern locally instead, per the task brief.
"""

from __future__ import annotations

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
        if k not in self.d:
            raise KeyError(k)
        return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def _deps(tmp_path, settings: Settings) -> IngestDeps:
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    ingestor = IncrementalIngestor(FakeEmbedder(), InMemoryVectorStore(),
                                   BM25Retriever(), JsonlManifestStore(str(tmp_path)))
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1_000_000)
    return IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=settings)


def _seed(deps, body: bytes, did="doc1", tenant="t1") -> None:
    deps.blobs.put(f"{tenant}/{did}", body)
    deps.registry.create(DocumentRecord(
        document_id=did, tenant_id=tenant, filename="f.txt", content_type="text/plain",
        size_bytes=len(body), status=DocumentStatus.PROCESSING, blob_key=f"{tenant}/{did}"))


# One long paragraph (no blank lines) of 2000 distinct words -- big enough
# that both the default and a much tighter token budget must split it into
# multiple chunks, so the counts are directly comparable.
_LONG_BODY = " ".join(f"word{i}" for i in range(2000)).encode()


def test_worker_chunk_config_changes_chunk_count(tmp_path):
    """A non-default chunk_max_tokens/chunk_overlap in Settings must change
    the chunks run_ingest (the API/worker path) produces -- proves
    ingest/worker.py no longer calls chunk_document(doc) bare with the
    hardcoded 256/32 signature defaults."""
    default_deps = _deps(tmp_path / "default", Settings(pii_mode="keep"))
    _seed(default_deps, _LONG_BODY)
    run_ingest(default_deps, "doc1")
    default_rec = default_deps.registry.get("doc1", "t1")
    assert default_rec.status == DocumentStatus.READY
    default_count = default_rec.chunk_count

    custom_deps = _deps(
        tmp_path / "custom",
        Settings(pii_mode="keep", chunk_max_tokens=10, chunk_overlap=2),
    )
    _seed(custom_deps, _LONG_BODY)
    run_ingest(custom_deps, "doc1")
    custom_rec = custom_deps.registry.get("doc1", "t1")
    assert custom_rec.status == DocumentStatus.READY
    custom_count = custom_rec.chunk_count

    assert default_count >= 1
    # A far smaller token budget must produce strictly more chunks of the
    # same document -- not just "a setting exists somewhere".
    assert custom_count > default_count
    assert len(custom_deps.ingestor._store.chunks) == custom_count
    assert len(default_deps.ingestor._store.chunks) == default_count
