"""Tests for the local test console (app/ui.py).

SECURITY: the console page and its dev-token endpoint must be completely
absent (404) unless `auth_dev_signer_enabled` is on — a flag `core/config.py`
refuses to accept when `app_env=prod`. These tests pin that fail-closed
behaviour, and prove the token handed to the browser is accepted by the real
`require_principal` verify path (not a bypass).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_pipeline
from app.auth import get_verifier
from core import config
from core.types import ACLContext
from providers.auth.jwt_verifier import JWTVerifier

UI_SECRET = "ui-console-secret"


class _FakePipeline:
    """Records the ACL every query ran under."""

    def __init__(self):
        self.calls: list[tuple[str, ACLContext]] = []

    def run(self, question, acl=None, *, collection_id=None):
        self.calls.append((question, acl))
        return {
            "answer": "canned",
            "citations": [],
            "retrieved_ids": [],
            "usage": {},
            "refused": False,
        }


@pytest.fixture()
def dev_signer_on(monkeypatch):
    """Enable the dev signer with a known secret, matching the verifier below."""
    config.get_settings.cache_clear()
    monkeypatch.setenv("AUTH_DEV_SIGNER_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", UI_SECRET)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture()
def dev_signer_off(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("AUTH_DEV_SIGNER_ENABLED", "false")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture()
def pipeline():
    return _FakePipeline()


@pytest.fixture()
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_verifier] = lambda: JWTVerifier(
        alg="HS256", hs_secret=UI_SECRET, max_acl_tags=32
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Fail-closed: no console, no minting surface, unless the dev signer is on
# ---------------------------------------------------------------------------


def test_console_page_is_404_when_dev_signer_disabled(client, dev_signer_off):
    assert client.get("/ui").status_code == 404


def test_token_endpoint_is_404_when_dev_signer_disabled(client, dev_signer_off):
    resp = client.post("/ui/token", json={"tenant_id": "tenant_a"})
    assert resp.status_code == 404


def test_token_endpoint_is_404_when_secret_missing(client, monkeypatch):
    """Flag on but no HS256 secret (e.g. an RS256 deploy) must not mint."""
    config.get_settings.cache_clear()
    monkeypatch.setenv("AUTH_DEV_SIGNER_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "")
    config.get_settings.cache_clear()
    resp = client.post("/ui/token", json={"tenant_id": "tenant_a"})
    assert resp.status_code == 404
    config.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Enabled behaviour
# ---------------------------------------------------------------------------


def test_console_page_is_served_when_enabled(client, dev_signer_on):
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<!doctype html" in resp.text.lower()


def test_minted_token_is_accepted_by_the_real_auth_path(client, dev_signer_on, pipeline):
    """The token the browser receives must work on /query — proving the console
    drives the same verified-JWT path as any other client, with no bypass."""
    minted = client.post(
        "/ui/token", json={"tenant_id": "tenant_a", "acl_tags": ["finance"]}
    )
    assert minted.status_code == 200
    token = minted.json()["token"]

    resp = client.post(
        "/query",
        json={"question": "who?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    _, acl = pipeline.calls[-1]
    assert acl.tenant_id == "tenant_a"
    assert "finance" in acl.acl_tags


def test_token_is_scoped_to_the_requested_tenant(client, dev_signer_on, pipeline):
    """Two tenants get distinct tokens that carry their own identity."""
    for tenant in ("tenant_a", "tenant_b"):
        token = client.post("/ui/token", json={"tenant_id": tenant}).json()["token"]
        client.post(
            "/query",
            json={"question": "q"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert pipeline.calls[-1][1].tenant_id == tenant


def test_token_endpoint_rejects_empty_tenant(client, dev_signer_on):
    assert client.post("/ui/token", json={"tenant_id": ""}).status_code == 422
