from fastapi.testclient import TestClient

from app.api import app, get_pipeline
from app.auth import require_principal
from core.types import Principal


class _Pipe:
    def __init__(self):
        self.seen = None

    def run(self, question, acl, *, collection_id=None):
        self.seen = collection_id
        return {"answer": "ok", "citations": [], "retrieved_ids": [], "usage": {}, "refused": False}


def test_query_forwards_collection_id():
    pipe = _Pipe()
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")
    try:
        c = TestClient(app)
        r = c.post("/query", json={"question": "hi", "collection_id": "A"})
        assert r.status_code == 200
        assert pipe.seen == "A"
    finally:
        app.dependency_overrides.clear()


def test_query_without_collection_id_defaults_none():
    pipe = _Pipe()
    pipe.seen = "sentinel"
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")
    try:
        c = TestClient(app)
        r = c.post("/query", json={"question": "hi"})
        assert r.status_code == 200
        assert pipe.seen is None
    finally:
        app.dependency_overrides.clear()
