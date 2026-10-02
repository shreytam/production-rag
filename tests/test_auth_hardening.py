"""Auth hardening: fail-closed defaults, claim typing, loopback-only dev signer."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api import app
from core import config
from core.config import Settings
from core.interfaces import AuthError
from providers.auth.dev_signer import mint_token
from providers.auth.jwt_verifier import JWTVerifier

SECRET = "x" * 40


def _prod(**kw):
    base = dict(
        app_env="prod",
        jwt_alg="HS256",
        jwt_secret=SECRET,
        jwt_issuer="iss",
        jwt_audience="aud",
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_hs256_empty_secret_rejected_any_env():
    with pytest.raises(ValueError):
        JWTVerifier(alg="HS256", hs_secret="")


@pytest.mark.parametrize(
    "secret",
    ["change-me-dev-secret", "dev-console-secret-not-for-production!", "change-me-x"],
)
def test_placeholder_secret_rejected_outside_dev(secret):
    with pytest.raises(ValueError):
        _prod(jwt_secret=secret)


def test_placeholder_secret_allowed_in_dev():
    Settings(_env_file=None, app_env="dev", jwt_secret="change-me-dev-secret")


def test_short_hs_secret_rejected_in_prod():
    with pytest.raises(ValueError):
        _prod(jwt_secret="short-secret")


def test_dev_signer_rejected_in_prod():
    with pytest.raises(ValueError):
        _prod(auth_dev_signer_enabled=True)


def test_good_prod_config_accepted():
    _prod()


def _tok(**claims):
    payload = {"exp": int(time.time()) + 600, **claims}
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_requires_iss_and_aud_when_configured():
    v = JWTVerifier(alg="HS256", hs_secret=SECRET, issuer="iss", audience="aud")
    with pytest.raises(AuthError):
        v.verify(_tok(tenant_id="t"))  # no iss/aud


def test_acl_tags_string_rejected():
    v = JWTVerifier(alg="HS256", hs_secret=SECRET)
    with pytest.raises(AuthError) as e:
        v.verify(_tok(tenant_id="t", acl_tags="admin"))
    assert e.value.status in (401, 403)


@pytest.mark.parametrize("tid", [123, ["a"], {"a": 1}, "  ", True])
def test_tenant_id_must_be_nonempty_str(tid):
    v = JWTVerifier(alg="HS256", hs_secret=SECRET)
    with pytest.raises(AuthError):
        v.verify(_tok(tenant_id=tid))


def test_acl_tags_non_str_items_rejected():
    v = JWTVerifier(alg="HS256", hs_secret=SECRET)
    with pytest.raises(AuthError):
        v.verify(_tok(tenant_id="t", acl_tags=["a", 1]))


def test_valid_claims_ok():
    v = JWTVerifier(alg="HS256", hs_secret=SECRET)
    p = v.verify(_tok(tenant_id="t", acl_tags=["a", "b"]))
    assert p.tenant_id == "t" and p.acl_tags == ("a", "b")


def test_mint_roundtrip():
    v = JWTVerifier(alg="HS256", hs_secret=SECRET)
    assert v.verify(mint_token(tenant_id="t", secret=SECRET)).tenant_id == "t"


@pytest.fixture()
def signer_env(monkeypatch):
    monkeypatch.setenv("AUTH_DEV_SIGNER_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "ui-console-secret")
    monkeypatch.delenv("AUTH_DEV_SIGNER_ALLOW_REMOTE", raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_ui_token_refuses_non_loopback_by_default(signer_env):
    c = TestClient(app)  # client host is "testclient"
    assert c.post("/ui/token", json={"tenant_id": "t"}).status_code == 404
    assert c.get("/ui").status_code == 404


def test_ui_token_allows_loopback(signer_env):
    c = TestClient(app, client=("127.0.0.1", 5000))
    assert c.post("/ui/token", json={"tenant_id": "t"}).status_code == 200


def test_ui_token_remote_opt_in(signer_env, monkeypatch):
    monkeypatch.setenv("AUTH_DEV_SIGNER_ALLOW_REMOTE", "true")
    config.get_settings.cache_clear()
    assert TestClient(app).post("/ui/token", json={"tenant_id": "t"}).status_code == 200


def test_qdrant_api_key_default_empty():
    assert Settings(_env_file=None).qdrant_api_key == ""
