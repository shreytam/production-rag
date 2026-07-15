"""Offline tests for app.api (FastAPI) and a smoke import of app.demo.

The real RAGPipeline is never constructed — we override the get_pipeline
dependency with a fake that returns canned data and records the ACL it
received, letting us verify tenant isolation plumbing without any
network/model/API-key access.

Security: identity (tenant_id/acl_tags) comes ONLY from a verified JWT — the
`get_verifier` dependency is overridden with a known-secret HS256 verifier so
tests can mint tokens; there is no client-controlled header/body identity path.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_pipeline
from app.auth import get_verifier
from core.types import ACLContext
from providers.auth.jwt_verifier import JWTVerifier
from providers.auth.dev_signer import mint_token

TEST_SECRET = "app-test-secret"


def _auth_header(tenant_id="tenant_a", acl_tags=()):
    token = mint_token(tenant_id=tenant_id, acl_tags=list(acl_tags), secret=TEST_SECRET)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------

CANNED_ANSWER = "Paris is the capital of France."
CANNED_CITATIONS = [{"marker": "[1]", "chunk_id": "c1", "doc_id": "d1", "quote": "Paris"}]
CANNED_RETRIEVED_IDS = ["d1"]
CANNED_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0}


class FakePipeline:
    """Records every (question, acl) call; returns canned data."""

    def __init__(self):
        self.calls: list[tuple[str, ACLContext]] = []

    def run(self, question: str, acl: ACLContext | None = None) -> dict[str, Any]:
        self.calls.append((question, acl))
        return {
            "answer": CANNED_ANSWER,
            "citations": CANNED_CITATIONS,
            "retrieved_ids": CANNED_RETRIEVED_IDS,
            "retrieved_chunk_ids": ["c1"],
            "contexts": ["Paris is the capital of France."],
            "usage": CANNED_USAGE,
            "refused": False,
        }


@pytest.fixture()
def fake_pipeline():
    return FakePipeline()


@pytest.fixture()
def client(fake_pipeline):
    """TestClient with the pipeline and verifier dependencies overridden."""
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_verifier] = lambda: JWTVerifier(
        alg="HS256", hs_secret=TEST_SECRET, max_acl_tags=32
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------

def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /query — happy path (requires a verified token)
# ---------------------------------------------------------------------------

def test_query_returns_200_with_answer_and_citations(client, fake_pipeline):
    resp = client.post("/query", json={"question": "What is the capital of France?"},
                       headers=_auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == CANNED_ANSWER
    assert data["citations"] == CANNED_CITATIONS
    assert data["retrieved_ids"] == CANNED_RETRIEVED_IDS
    assert data["refused"] is False


# ---------------------------------------------------------------------------
# Auth failures
# ---------------------------------------------------------------------------

def test_query_without_token_is_401(client):
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_query_bad_signature_is_401(client):
    bad = mint_token(tenant_id="tenant_a", secret="not-the-secret")
    resp = client.post("/query", json={"question": "hi"},
                       headers={"Authorization": f"Bearer {bad}"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Identity source: token only, never a spoofed header/body
# ---------------------------------------------------------------------------

def test_identity_comes_only_from_verified_token(client, fake_pipeline):
    """A spoofed X-Tenant-Id header and any body identity are ignored; the ACL the
    pipeline receives comes from the signed token."""
    headers = _auth_header(tenant_id="tenant_a", acl_tags=("finance",))
    headers["X-Tenant-Id"] = "tenant_b"  # attacker attempt
    resp = client.post("/query", json={"question": "hi", "tenant_id": "tenant_b"},
                       headers=headers)
    assert resp.status_code == 200
    _, acl = fake_pipeline.calls[0]
    assert acl.tenant_id == "tenant_a"        # from the token, NOT the header/body
    assert set(acl.acl_tags) == {"finance"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_oversized_question_is_422(client, monkeypatch):
    from core import config
    config.get_settings.cache_clear()
    monkeypatch.setenv("MAX_QUESTION_CHARS", "10")
    config.get_settings.cache_clear()
    resp = client.post("/query", json={"question": "x" * 50}, headers=_auth_header())
    assert resp.status_code == 422
    config.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Smoke import of demo module (no Streamlit server needed)
# ---------------------------------------------------------------------------

def test_demo_imports_cleanly():
    """app.demo must be importable without a running backend or Streamlit."""
    import app.demo  # noqa: F401


def test_demo_principal_roundtrip(monkeypatch):
    """The demo's auth helper mints + verifies a token, yielding an ACL scoped to
    the selected org — exercising the real verify path, not a raw dropdown value."""
    from core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("AUTH_DEV_SIGNER_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "demo-secret")
    config.get_settings.cache_clear()

    from app.auth import demo_principal

    p = demo_principal("tenant_a", acl_tags=("finance",))
    assert p.to_acl().tenant_id == "tenant_a"
    assert set(p.acl_tags) == {"finance"}

    config.get_settings.cache_clear()
