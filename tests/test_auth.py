from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_auth_defaults_are_dev_friendly():
    s = Settings()
    assert s.auth_enabled is True
    assert s.app_env == "dev"
    assert s.jwt_alg == "HS256"
    assert s.auth_dev_signer_enabled is False
    assert s.max_question_chars == 8000
    assert s.max_acl_tags == 32


def test_prod_requires_hs256_secret():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="",
                 jwt_issuer="iss", jwt_audience="aud")


def test_prod_requires_issuer_and_audience():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="s",
                 jwt_issuer="", jwt_audience="")


def test_prod_forbids_dev_signer():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="s",
                 jwt_issuer="iss", jwt_audience="aud", auth_dev_signer_enabled=True)


def test_prod_valid_config_constructs():
    s = Settings(app_env="prod", jwt_alg="RS256", jwks_url="https://idp/jwks",
                 jwt_issuer="iss", jwt_audience="aud")
    assert s.app_env == "prod"


from core.types import ACLContext, Principal
from core.interfaces import AuthError, AuthVerifier, TenantAllowlist


def test_principal_to_acl_maps_tenant_and_tags():
    p = Principal(tenant_id="acme", acl_tags=("finance",), subject="u1", claims={"sub": "u1"})
    acl = p.to_acl()
    assert isinstance(acl, ACLContext)
    assert acl.tenant_id == "acme"
    assert acl.acl_tags == ("finance",)


def test_principal_rejects_blank_tenant():
    with pytest.raises(Exception):
        Principal(tenant_id="  ", acl_tags=())


def test_principal_is_frozen():
    p = Principal(tenant_id="acme")
    with pytest.raises(Exception):
        p.tenant_id = "other"


def test_aclcontext_rejects_blank_tenant():
    with pytest.raises(Exception):
        ACLContext(tenant_id="")


def test_autherror_carries_status():
    e = AuthError("nope", status=403)
    assert e.status == 403
    assert e.detail == "nope"
    assert str(e) == "nope"


import jwt as _jwt  # pyjwt

from providers.auth.dev_signer import mint_token


def test_mint_token_roundtrips_claims():
    token = mint_token(tenant_id="acme", acl_tags=["finance", "hr"], secret="s3cret",
                       issuer="test-iss", audience="test-aud", ttl_seconds=120)
    decoded = _jwt.decode(token, "s3cret", algorithms=["HS256"],
                          issuer="test-iss", audience="test-aud")
    assert decoded["tenant_id"] == "acme"
    assert decoded["acl_tags"] == ["finance", "hr"]
    assert "exp" in decoded and "nbf" in decoded and "iat" in decoded
