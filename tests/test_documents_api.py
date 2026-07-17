import pytest
from fastapi.testclient import TestClient

from app.api import app
from app import documents as docs_mod
from core.types import Principal
from ingest.parsers.base import ParserRegistry
from providers.docstore.memory import InMemoryDocumentRegistry


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k): return self.d[k]
    def delete(self, k): self.d.pop(k, None)


@pytest.fixture
def client():
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers

    async def fake_enqueue(document_id): enqueued.append(document_id)
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue

    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    c = TestClient(app)
    c.enqueued = enqueued
    c.registry = reg
    yield c
    app.dependency_overrides.clear()


def test_upload_returns_202_and_enqueues(client):
    r = client.post("/documents", files={"file": ("n.txt", b"hello", "text/plain")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing"
    assert client.enqueued == [body["document_id"]]


def test_upload_rejects_disallowed_type(client):
    r = client.post("/documents", files={"file": ("n.bin", b"x", "application/x-nope")})
    assert r.status_code == 415


def test_upload_rejects_oversize(client):
    big = b"x" * 2000  # parsers max_bytes=1000
    r = client.post("/documents", files={"file": ("n.txt", big, "text/plain")})
    assert r.status_code == 413


def test_get_status_is_tenant_scoped(client):
    r = client.post("/documents", files={"file": ("n.txt", b"hi", "text/plain")})
    did = r.json()["document_id"]
    # Same tenant sees it
    assert client.get(f"/documents/{did}").status_code == 200
    # Switch principal to another tenant -> 404
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="other")
    assert client.get(f"/documents/{did}").status_code == 404


def test_upload_stores_collection_id(client):
    r = client.post("/documents",
                    data={"collection_id": "projX"},
                    files={"file": ("n.txt", b"hi", "text/plain")})
    assert r.status_code == 202
    did = r.json()["document_id"]
    assert client.registry.get(did, "t1").collection_id == "projX"


def test_list_documents_tenant_scoped_and_filterable(client):
    a = client.post("/documents", data={"collection_id": "A"},
                    files={"file": ("a.txt", b"a", "text/plain")}).json()["document_id"]
    client.post("/documents", data={"collection_id": "B"},
                files={"file": ("b.txt", b"b", "text/plain")})
    all_docs = client.get("/documents").json()
    assert {d["document_id"] for d in all_docs} >= {a}
    only_a = client.get("/documents", params={"collection_id": "A"}).json()
    assert [d["document_id"] for d in only_a] == [a]
    assert all(d["collection_id"] == "A" for d in only_a)

    # Switch principal to another tenant -> t1's documents must not leak
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="other")
    assert client.get("/documents").json() == []
    assert client.get("/documents", params={"collection_id": "A"}).json() == []


def test_upload_rejects_bad_collection_id(client):
    r = client.post("/documents", data={"collection_id": "x" * 200},
                    files={"file": ("n.txt", b"hi", "text/plain")})
    assert r.status_code == 422
