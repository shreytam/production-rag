import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from app import documents as docs_mod
from app import ratelimit
from app.api import app, get_pipeline
from app.auth import require_principal
from core.config import get_settings
from core.types import DocumentStatus, Principal
from ingest.parsers.base import ParserError, ParserRegistry
from providers.docstore.memory import InMemoryDocumentRegistry

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DictBlobs:
    def __init__(self):
        self.d = {}
        self.put_thread = None

    def put(self, k, v):
        self.put_thread = threading.get_ident()
        self.d[k] = v

    def get(self, k): return self.d[k]
    def delete(self, k): self.d.pop(k, None)


@pytest.fixture
def env():
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(
        allowed_types={"text/plain", "application/pdf", DOCX}, max_bytes=3 * 1024 * 1024
    )
    state = {"fail": False, "calls": [], "loop_thread": None}

    async def enqueue(document_id, action="ingest"):
        state["loop_thread"] = threading.get_ident()
        if state["fail"]:
            raise ConnectionError("redis down")
        state["calls"].append((document_id, action))

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: enqueue
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")
    c = TestClient(app, raise_server_exceptions=False)
    yield c, reg, blobs, state
    app.dependency_overrides.clear()


def _up(c, body=b"hi", ctype="text/plain", name="a.txt"):
    return c.post("/documents", files={"file": (name, body, ctype)})


# --- magic-byte sniffing -------------------------------------------------


def test_pdf_declared_but_not_pdf_is_415(env):
    assert _up(env[0], b"not a pdf", "application/pdf").status_code == 415


def test_real_pdf_header_accepted(env):
    assert _up(env[0], b"%PDF-1.7 body", "application/pdf").status_code == 202


def test_docx_must_be_zip(env):
    c = env[0]
    assert _up(c, b"plain", DOCX).status_code == 415
    assert _up(c, b"PK\x03\x04rest", DOCX).status_code == 202


def test_text_must_be_utf8(env):
    assert _up(env[0], b"\xff\xfe\x00bad").status_code == 415


def test_validate_content_unit():
    p = ParserRegistry(allowed_types={"application/pdf"}, max_bytes=10)
    with pytest.raises(ParserError):
        p.validate_content("application/pdf", b"zzz")
    p.validate_content("application/pdf", b"%PDF-1")


# --- streaming size cap --------------------------------------------------


def test_oversize_aborts_without_unbounded_read(env, monkeypatch):
    c, _, blobs, _ = env
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=2 * 1024 * 1024)
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    reads = []
    from starlette.datastructures import UploadFile as _SU
    orig = _SU.read

    async def spy(self, size=-1):
        reads.append(size)
        return await orig(self, size)

    monkeypatch.setattr(_SU, "read", spy)
    r = _up(c, b"x" * (5 * 1024 * 1024))
    assert r.status_code == 413
    assert reads and all(s > 0 for s in reads)
    assert blobs.d == {}


def test_content_length_over_cap_rejected_early(env, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 100)
    assert _up(env[0], b"x" * 5000).status_code == 413


# --- enqueue failures ----------------------------------------------------


def test_upload_enqueue_failure_503_and_no_stranded_state(env):
    c, reg, blobs, state = env
    state["fail"] = True
    assert _up(c).status_code == 503
    assert reg.list("t1") == []
    assert blobs.d == {}


def test_delete_enqueue_failure_reverts_status_and_503(env):
    c, reg, _, state = env
    did = _up(c).json()["document_id"]
    reg.set_status(did, "t1", DocumentStatus.READY, chunk_count=3)
    state["fail"] = True
    assert c.delete(f"/documents/{did}").status_code == 503
    rec = reg.get(did, "t1")
    assert rec.status == DocumentStatus.READY and rec.chunk_count == 3


def test_delete_while_processing_is_409(env):
    c, reg, _, state = env
    did = _up(c).json()["document_id"]
    state["calls"].clear()
    assert c.delete(f"/documents/{did}").status_code == 409
    assert reg.get(did, "t1").status == DocumentStatus.PROCESSING
    assert state["calls"] == []


# --- event loop not blocked ----------------------------------------------


def test_blob_put_runs_off_event_loop(env):
    c, _, blobs, state = env
    assert _up(c).status_code == 202
    assert blobs.put_thread != state["loop_thread"]


# --- rate limiting -------------------------------------------------------


@pytest.fixture
def limited(env, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "rate_limit_enabled", True)
    monkeypatch.setattr(s, "rate_limit_upload_per_minute", 2)
    monkeypatch.setattr(s, "rate_limit_query_per_minute", 2)
    lim = ratelimit.MemoryRateLimiter()
    app.dependency_overrides[ratelimit.get_limiter] = lambda: lim
    return env


def test_upload_rate_limited_429_with_retry_after(limited):
    c = limited[0]
    for _ in range(2):
        assert _up(c).status_code == 202
    r = _up(c)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_rate_limit_is_per_tenant(limited):
    c = limited[0]
    for _ in range(2):
        _up(c)
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t2")
    assert _up(c).status_code == 202


def test_query_rate_limited_separately(limited):
    c = limited[0]

    class P:
        def run(self, q, acl, collection_id=None):
            return {"answer": "a"}

    app.dependency_overrides[get_pipeline] = lambda: P()
    for _ in range(2):
        assert c.post("/query", json={"question": "q"}).status_code == 200
    assert c.post("/query", json={"question": "q"}).status_code == 429
    assert _up(c).status_code == 202  # upload budget untouched


def test_rate_limit_disabled(limited, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_enabled", False)
    for _ in range(5):
        assert _up(limited[0]).status_code == 202


def test_memory_limiter_window_resets():
    t = [0.0]
    lim = ratelimit.MemoryRateLimiter(clock=lambda: t[0])
    assert asyncio.run(lim.hit("k", 1)) == 0
    assert asyncio.run(lim.hit("k", 1)) > 0
    t[0] = 61
    assert asyncio.run(lim.hit("k", 1)) == 0


def test_redis_limiter_fails_open(caplog):
    lim = ratelimit.RedisRateLimiter("redis://127.0.0.1:1")
    with caplog.at_level("WARNING"):
        assert asyncio.run(lim.hit("k", 1)) == 0
    assert "failing open" in caplog.text
