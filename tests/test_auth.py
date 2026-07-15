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


import json
import time

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from providers.auth.jwt_verifier import JWTVerifier

HS_SECRET = "unit-test-secret"


def _hs_verifier(**kw):
    return JWTVerifier(alg="HS256", hs_secret=HS_SECRET, **kw)


def test_verify_valid_hs256_returns_principal():
    token = mint_token(tenant_id="acme", acl_tags=["finance"], secret=HS_SECRET)
    p = _hs_verifier().verify(token)
    assert p.tenant_id == "acme"
    assert p.acl_tags == ("finance",)
    assert p.subject == "dev"


def test_verify_bad_signature_rejected():
    token = mint_token(tenant_id="acme", secret="the-wrong-secret")
    with pytest.raises(AuthError) as ei:
        _hs_verifier().verify(token)
    assert ei.value.status == 401


def test_verify_expired_rejected():
    # NOTE: brief specified ttl_seconds=-10, but JWTVerifier's default
    # leeway_seconds=60 (matching Settings.jwt_leeway_seconds, also 60 — see
    # core/config.py) legitimately tolerates a token that is only 10s past
    # expiry as clock skew. Using -120 (past the leeway window) keeps this test
    # a genuine test of expiry rejection rather than a false negative caused by
    # the leeway grace period.
    token = mint_token(tenant_id="acme", secret=HS_SECRET, ttl_seconds=-120)
    with pytest.raises(AuthError) as ei:
        _hs_verifier().verify(token)
    assert ei.value.status == 401


def test_verify_alg_none_rejected():
    token = _jwt.encode({"tenant_id": "acme", "exp": int(time.time()) + 60,
                         "iat": int(time.time()), "nbf": int(time.time())},
                        key=None, algorithm="none")
    with pytest.raises(AuthError):
        _hs_verifier().verify(token)


def test_verify_missing_tenant_is_403():
    token = _jwt.encode({"sub": "u", "exp": int(time.time()) + 60,
                         "iat": int(time.time()), "nbf": int(time.time())}, HS_SECRET,
                        algorithm="HS256")
    with pytest.raises(AuthError) as ei:
        _hs_verifier().verify(token)
    assert ei.value.status == 403


def test_verify_too_many_tags_is_403():
    token = mint_token(tenant_id="acme", acl_tags=[f"t{i}" for i in range(5)], secret=HS_SECRET)
    with pytest.raises(AuthError) as ei:
        _hs_verifier(max_acl_tags=3).verify(token)
    assert ei.value.status == 403


def test_verify_audience_mismatch_rejected():
    token = mint_token(tenant_id="acme", secret=HS_SECRET, audience="aud-a")
    with pytest.raises(AuthError):
        _hs_verifier(audience="aud-b").verify(token)


# --- RS256 / JWKS + algorithm confusion ---

def _rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def _jwks_for(pub_pem: str, kid: str) -> dict:
    numbers = serialization.load_pem_public_key(pub_pem.encode()).public_numbers()
    import base64

    def b64u(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    return {"keys": [{"kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256",
                      "n": b64u(numbers.n), "e": b64u(numbers.e)}]}


def test_verify_valid_rs256_via_jwks():
    priv, pub = _rsa_keypair()
    kid = "key-1"
    jwks = _jwks_for(pub, kid)
    token = _jwt.encode({"tenant_id": "acme", "acl_tags": ["hr"],
                         "iat": int(time.time()), "nbf": int(time.time()),
                         "exp": int(time.time()) + 60}, priv,
                        algorithm="RS256", headers={"kid": kid})
    v = JWTVerifier(alg="RS256", jwks_fetcher=lambda url: jwks)
    p = v.verify(token)
    assert p.tenant_id == "acme"
    assert p.acl_tags == ("hr",)


def _forge_hs256_with_pem_secret(payload: dict, pem_secret: str, kid: str) -> str:
    """Hand-build an HS256 JWS using a PEM string as the raw HMAC secret.

    pyjwt >= 2.10 refuses to `encode()` an HS256 token when the key looks like a
    PEM/asymmetric key (it detects this exact attack at construction time). A real
    attacker forging tokens is not bound by pyjwt's guard rails, so we build the
    JWS by hand — signing_input = base64url(header).base64url(payload), HMAC-SHA256
    with the PEM string as key — to faithfully exercise whether OUR verifier's
    algorithm pinning (not pyjwt's encode-side heuristic) rejects the forgery.
    """
    import base64
    import hashlib
    import hmac

    def b64u(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    signing_input = (
        f"{b64u(json.dumps(header, separators=(',', ':')).encode())}."
        f"{b64u(json.dumps(payload, separators=(',', ':')).encode())}"
    ).encode()
    sig = hmac.new(pem_secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{signing_input.decode()}.{b64u(sig)}"


def test_algorithm_confusion_hs_forged_with_rsa_pubkey_rejected():
    """Classic attack: forge an HS256 token using the RSA PUBLIC key as the HMAC
    secret. A verifier pinned to RS256 must reject it."""
    priv, pub = _rsa_keypair()
    kid = "key-1"
    jwks = _jwks_for(pub, kid)
    forged = _forge_hs256_with_pem_secret(
        {"tenant_id": "attacker", "iat": int(time.time()),
         "nbf": int(time.time()), "exp": int(time.time()) + 60},
        pub, kid,
    )
    v = JWTVerifier(alg="RS256", jwks_fetcher=lambda url: jwks)
    with pytest.raises(AuthError):
        v.verify(forged)


# --- Allowlist Tests ---

from providers.auth.allowlist import NullAllowlist, StaticAllowlist, apply_allowlist


def test_null_allowlist_passes_tags_through():
    al = NullAllowlist()
    assert al.allowed("acme") is None
    assert apply_allowlist(al, "acme", ("finance", "hr")) == ("finance", "hr")


def test_static_allowlist_intersects():
    al = StaticAllowlist({"acme": ["finance"]})
    assert apply_allowlist(al, "acme", ("finance", "hr")) == ("finance",)


def test_static_allowlist_unknown_tenant_drops_all_tags():
    al = StaticAllowlist({"acme": ["finance"]})
    assert apply_allowlist(al, "globex", ("finance",)) == ()


def test_static_allowlist_from_file(tmp_path):
    p = tmp_path / "acl.json"
    p.write_text(json.dumps({"acme": ["finance", "hr"]}))
    al = StaticAllowlist.from_file(str(p))
    assert apply_allowlist(al, "acme", ("finance", "legal")) == ("finance",)
