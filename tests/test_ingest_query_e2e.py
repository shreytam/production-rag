"""End-to-end integration proof: upload -> ingest worker -> query retrieves it.

This exercises the real API router, the pure `run_ingest` worker body, the
`IncrementalIngestor`, and the `HybridRetriever` over a shared set of offline
fakes. It owns no production code; it guards the seams between the ingest and
query paths and the tenant-isolation boundary.
"""

from fastapi.testclient import TestClient

from app import documents as docs_mod
from app.api import app
from core.config import Settings
from core.types import ACLContext, Principal, Query
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k): return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def test_uploaded_document_is_retrievable(tmp_path):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=10000)
    store = InMemoryVectorStore()
    sparse = BM25Retriever()
    ingestor = IncrementalIngestor(FakeEmbedder(), store, sparse,
                                   JsonlManifestStore(str(tmp_path)))
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    async def fake_enqueue(document_id): enqueued.append(document_id)
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    try:
        client = TestClient(app)
        body = b"the quick brown fox jumps over the lazy dog"
        r = client.post("/documents", files={"file": ("f.txt", body, "text/plain")})
        did = r.json()["document_id"]

        # Run the worker synchronously against the SAME fakes.
        deps = IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                          ingestor=ingestor, settings=Settings(pii_mode="keep"))
        run_ingest(deps, did)
        assert client.get(f"/documents/{did}").json()["status"] == "ready"

        # Retrieve via the hybrid path over the same store+sparse.
        retriever = HybridRetriever(FakeEmbedder(), store, sparse, FakeReranker())
        hits = retriever.retrieve(Query(text="quick brown fox",
                                        acl=ACLContext(tenant_id="t1"), top_k=5, rerank_top_n=3))
        assert any(h.chunk.doc_id == did for h in hits)

        # Isolation: another tenant sees nothing.
        empty = retriever.retrieve(Query(text="quick brown fox",
                                         acl=ACLContext(tenant_id="t2"), top_k=5, rerank_top_n=3))
        assert empty == []
    finally:
        app.dependency_overrides.clear()
