from fastapi.testclient import TestClient

from app import documents as docs_mod
from app.api import app
from core.config import Settings
from core.types import ACLContext, Principal, Query
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_delete, run_ingest
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


def test_collection_scoping_and_delete(tmp_path):
    reg, blobs = InMemoryDocumentRegistry(), DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=10000)
    store, sparse = InMemoryVectorStore(), BM25Retriever()
    ingestor = IncrementalIngestor(FakeEmbedder(), store, sparse, JsonlManifestStore(str(tmp_path)))
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    async def fake_enqueue(document_id, action="ingest"): enqueued.append((document_id, action))
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    deps = IngestDeps(registry=reg, blobs=blobs, parsers=parsers, ingestor=ingestor,
                      settings=Settings(pii_mode="keep"))
    try:
        client = TestClient(app)
        a = client.post("/documents", data={"collection_id": "A"},
                        files={"file": ("a.txt", b"the quick brown fox", "text/plain")}).json()["document_id"]
        b = client.post("/documents", data={"collection_id": "B"},
                        files={"file": ("b.txt", b"the quick brown fox", "text/plain")}).json()["document_id"]
        run_ingest(deps, a)
        run_ingest(deps, b)

        retriever = HybridRetriever(FakeEmbedder(), store, sparse, FakeReranker())
        hits = retriever.retrieve(Query(text="quick brown fox", acl=ACLContext(tenant_id="t1"),
                                        top_k=10, rerank_top_n=5, collection_id="A"))
        assert {h.chunk.doc_id for h in hits} == {a}

        # Delete A, then it disappears from retrieval and from the list.
        assert client.delete(f"/documents/{a}").status_code == 202
        run_delete(deps, a)
        hits2 = retriever.retrieve(Query(text="quick brown fox", acl=ACLContext(tenant_id="t1"),
                                         top_k=10, rerank_top_n=5, collection_id="A"))
        assert hits2 == []
        listed = client.get("/documents").json()
        assert a not in {d["document_id"] for d in listed}
        assert b in {d["document_id"] for d in listed}
    finally:
        app.dependency_overrides.clear()
