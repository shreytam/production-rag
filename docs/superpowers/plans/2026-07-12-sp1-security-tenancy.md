# SP1 · Security & Tenancy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the spoofable header/body tenant identity on `/query` with cryptographically verified JWT-derived identity, so every request's `tenant_id`/`acl_tags` come only from a signed token.

**Architecture:** A pluggable `AuthVerifier` Protocol (first impl `JWTVerifier`) turns a Bearer token into a verified `Principal`; an optional `TenantAllowlist` intersects claimed tags against per-tenant grants; a FastAPI dependency (`require_principal`) enforces this on `/query` and builds the `ACLContext` from the verified principal. Concrete classes are named only in `core/registry.py`, matching the existing codebase pattern. HS256 for self-contained dev, RS256+JWKS for prod.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, `pyjwt[crypto]` (JWT verify + RSA), httpx (JWKS fetch), pytest.

## Global Constraints

- **Commit authorship:** every commit authored solely as `Shreytam Goyal <shreytamgoyal@gmail.com>`. NO `Co-Authored-By:`, `Claude-Session:`, or "Generated with Claude" trailers — in commit messages or anywhere. (Repo `CLAUDE.md`.)
- **Algorithm pinning:** JWT decode MUST pass `algorithms=[settings.jwt_alg]` — a single-element list, never derived from the token's `alg` header. This is a hard security requirement.
- **Fail closed:** any verification failure (missing/malformed/expired/bad-signature token, missing/blank tenant claim, JWKS fetch failure) results in rejection (`401`/`403`), never a pass-through.
- **No secret logging:** tokens, secrets, and keys are never logged.
- **Identity source:** `tenant_id`/`acl_tags` derive ONLY from verified claims. No client-controlled header/body identity path may remain.
- **TDD:** every task writes the failing test first, watches it fail, then implements. Frequent commits.
- **Python floor:** `requires-python = ">=3.11,<3.14"`; `pyjwt[crypto]>=2.9`.

---

## File Structure

**Create:**
- `providers/auth/__init__.py` — package marker
- `providers/auth/dev_signer.py` — `mint_token(...)` HS256 signing helper (dev/demo/CLI)
- `providers/auth/jwt_verifier.py` — `JWTVerifier` (+ `_JWKSCache`)
- `providers/auth/allowlist.py` — `NullAllowlist`, `StaticAllowlist`, `apply_allowlist(...)`
- `app/auth.py` — `require_principal`, `get_verifier`, `get_allowlist` FastAPI dependencies + `demo_principal`
- `scripts/mint_token.py` — CLI wrapper around the dev signer
- `tests/test_auth.py` — unit tests for config validation, Principal, verifier, allowlist, registry, dev signer

**Modify:**
- `core/config.py` — 12 auth knobs + prod boot-validation
- `core/types.py` — `Principal` model, `ACLContext.tenant_id` non-empty validator
- `core/interfaces.py` — `AuthError`, `AuthVerifier` Protocol, `TenantAllowlist` Protocol
- `core/registry.py` — `build_auth_verifier`, `build_allowlist`
- `app/api.py` — `require_principal` dependency, `QueryRequest` loses identity fields, input cap, error mapping
- `app/demo.py` — mint+verify a token per selected org (real auth path)
- `pyproject.toml` — `pyjwt[crypto]` in `app` + `all` extras
- `infra/.env.example` — document new knobs
- `tests/test_app.py` — invert the spoof tests; add token-based tests

---

### Task 1: Auth config knobs + prod boot-validation

**Files:**
- Modify: `core/config.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces: `Settings` gains fields `auth_enabled: bool`, `app_env: Literal["dev","prod"]`, `jwt_alg: Literal["HS256","RS256"]`, `jwt_secret: str`, `jwks_url: str`, `jwt_issuer: str`, `jwt_audience: str`, `jwt_leeway_seconds: int`, `acl_allowlist_source: str`, `auth_dev_signer_enabled: bool`, `max_question_chars: int`, `max_acl_tags: int`. Constructing `Settings(app_env="prod", ...)` with an invalid auth config raises `ValidationError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k "auth_defaults or prod_" -v`
Expected: FAIL — `Settings` has no `auth_enabled`/`app_env` fields (or no prod validation).

- [ ] **Step 3: Add the knobs and validator**

In `core/config.py`, add these fields inside `class Settings` (place after the `# --- Guardrails ---` block, before `# --- Observability ---`):

```python
    # --- Auth & tenancy (SP1) ---
    auth_enabled: bool = True
    app_env: Literal["dev", "prod"] = "dev"
    jwt_alg: Literal["HS256", "RS256"] = "HS256"
    jwt_secret: str = ""          # HS256 shared secret (dev); presence enables minting
    jwks_url: str = ""            # RS256 JWKS endpoint (prod)
    jwt_issuer: str = ""          # enforced on prod
    jwt_audience: str = ""        # enforced on prod
    jwt_leeway_seconds: int = 60
    acl_allowlist_source: str = ""  # empty = NullAllowlist; path = StaticAllowlist JSON
    auth_dev_signer_enabled: bool = False
    max_question_chars: int = 8000
    max_acl_tags: int = 32
```

Add a new validator method to `class Settings`, immediately after the existing `_fill_key_fallbacks` method:

```python
    @model_validator(mode="after")
    def _validate_auth(self) -> "Settings":
        """Prod instances must be securely configured — fail fast at construction."""
        if self.app_env == "prod":
            if not self.auth_enabled:
                raise ValueError("auth_enabled must be True when app_env=prod")
            if self.jwt_alg == "HS256" and not self.jwt_secret:
                raise ValueError("HS256 requires jwt_secret when app_env=prod")
            if self.jwt_alg == "RS256" and not self.jwks_url:
                raise ValueError("RS256 requires jwks_url when app_env=prod")
            if not self.jwt_issuer or not self.jwt_audience:
                raise ValueError("jwt_issuer and jwt_audience are required when app_env=prod")
            if self.auth_dev_signer_enabled:
                raise ValueError("auth_dev_signer_enabled must be False when app_env=prod")
        return self
```

(`Literal` and `model_validator` are already imported at the top of `core/config.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k "auth_defaults or prod_" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_auth.py
git commit -m "Add SP1 auth config knobs and prod boot-validation"
```

---

### Task 2: Contracts — Principal, ACLContext validator, AuthError, Protocols

**Files:**
- Modify: `core/types.py`, `core/interfaces.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces:
  - `core.types.Principal(tenant_id: str, acl_tags: tuple[str,...] = (), subject: str = "", claims: dict = {})` — frozen; `tenant_id` must be non-empty; method `.to_acl() -> ACLContext`.
  - `core.types.ACLContext` — now rejects blank `tenant_id`.
  - `core.interfaces.AuthError(detail: str, status: int = 401)` — exception with `.detail` and `.status`.
  - `core.interfaces.AuthVerifier` Protocol: `verify(token: str) -> Principal`.
  - `core.interfaces.TenantAllowlist` Protocol: `allowed(tenant_id: str) -> frozenset[str] | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k "principal or aclcontext_rejects or autherror" -v`
Expected: FAIL — `Principal`/`AuthError`/`TenantAllowlist` not importable.

- [ ] **Step 3: Implement the contracts**

In `core/types.py`, update the import line:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
```

Add a non-empty validator to `class ACLContext` (after the `acl_tags` field, before `def allows`):

```python
    @field_validator("tenant_id")
    @classmethod
    def _acl_tenant_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ACLContext.tenant_id must be non-empty")
        return v
```

Add the `Principal` model directly below `class ACLContext`:

```python
class Principal(BaseModel):
    """A cryptographically verified caller identity. Built ONLY from a verified
    token's claims — never from raw request headers/body. `ACLContext` is derived
    from this, closing the tenant-spoofing hole."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    acl_tags: tuple[str, ...] = ()
    subject: str = ""
    claims: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def _tenant_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Principal.tenant_id must be non-empty")
        return v

    def to_acl(self) -> "ACLContext":
        return ACLContext(tenant_id=self.tenant_id, acl_tags=self.acl_tags)
```

In `core/interfaces.py`, add `Principal` to the `from core.types import (...)` block, then add near the top (after the imports):

```python
class AuthError(Exception):
    """Raised by an AuthVerifier when a token cannot be trusted.

    `status` is the HTTP status the API layer should surface: 401 (unauthenticated)
    or 403 (authenticated but no valid tenant / over-scoped token).
    """

    def __init__(self, detail: str, status: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status
```

And add the two Protocols (place near the other Protocols, e.g. after `Guardrail`):

```python
@runtime_checkable
class AuthVerifier(Protocol):
    """Turns a bearer token into a verified Principal. Raises AuthError on ANY
    failure — implementations must fail closed."""

    def verify(self, token: str) -> Principal: ...


@runtime_checkable
class TenantAllowlist(Protocol):
    """Per-tenant permitted acl_tags. `allowed()` returns None when unrestricted
    (claims pass through), else the frozenset of tags the tenant may hold."""

    def allowed(self, tenant_id: str) -> frozenset[str] | None: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k "principal or aclcontext_rejects or autherror" -v`
Expected: PASS (5 tests).

Then run the full suite to confirm the new `ACLContext` validator broke nothing:
Run: `uv run pytest -q`
Expected: PASS (all existing tests still green — no code constructs `ACLContext` with a blank tenant).

- [ ] **Step 5: Commit**

```bash
git add core/types.py core/interfaces.py tests/test_auth.py
git commit -m "Add Principal, ACLContext tenant validation, and auth contracts"
```

---

### Task 3: Dev token signer (+ pyjwt dependency)

**Files:**
- Create: `providers/auth/__init__.py`, `providers/auth/dev_signer.py`
- Modify: `pyproject.toml`
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces: `providers.auth.dev_signer.mint_token(*, tenant_id: str, acl_tags: Iterable[str] = (), subject: str = "dev", secret: str, issuer: str = "", audience: str = "", ttl_seconds: int = 3600, extra_claims: dict | None = None) -> str` — an HS256 JWT string with `iat`/`nbf`/`exp` set.

- [ ] **Step 1: Add pyjwt to pyproject**

In `pyproject.toml`, add `"pyjwt[crypto]>=2.9"` to BOTH the `app` extra and the `all` extra lists:

```toml
app = ["fastapi>=0.112", "uvicorn>=0.30", "streamlit>=1.37", "pyjwt[crypto]>=2.9"]
```
and append `"pyjwt[crypto]>=2.9",` to the `all = [...]` list.

Then install:
Run: `uv sync --all-extras`
Expected: resolves and installs `pyjwt` + `cryptography`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_auth.py  (append)
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k mint_token -v`
Expected: FAIL — `providers.auth.dev_signer` does not exist.

- [ ] **Step 4: Implement the signer**

Create `providers/auth/__init__.py`:

```python
"""Auth providers: JWT verification, tenant allowlists, and a dev token signer."""
```

Create `providers/auth/dev_signer.py`:

```python
"""Dev/demo HS256 token signer.

SECURITY: minting requires an HS256 `secret`. A prod instance runs RS256 with only
a public key and therefore CANNOT mint tokens, regardless of any flag. This helper
is for local development, the Streamlit demo, and the mint_token CLI only.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import jwt  # pyjwt


def mint_token(
    *,
    tenant_id: str,
    acl_tags: Iterable[str] = (),
    subject: str = "dev",
    secret: str,
    issuer: str = "",
    audience: str = "",
    ttl_seconds: int = 3600,
    extra_claims: dict | None = None,
) -> str:
    if not secret:
        raise ValueError("mint_token requires a non-empty HS256 secret")
    now = int(time.time())
    payload: dict = {
        "sub": subject,
        "tenant_id": tenant_id,
        "acl_tags": list(acl_tags),
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
    }
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k mint_token -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add providers/auth/__init__.py providers/auth/dev_signer.py pyproject.toml tests/test_auth.py
git commit -m "Add dev HS256 token signer and pyjwt dependency"
```

---

### Task 4: JWTVerifier (HS256 + RS256/JWKS, algorithm-pinned)

**Files:**
- Create: `providers/auth/jwt_verifier.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `mint_token` (Task 3), `AuthError`/`Principal` (Task 2).
- Produces: `providers.auth.jwt_verifier.JWTVerifier(*, alg: str, hs_secret: str = "", jwks_url: str = "", issuer: str = "", audience: str = "", leeway_seconds: int = 60, max_acl_tags: int = 32, jwks_fetcher: Callable[[str], dict] | None = None, jwks_ttl_seconds: int = 3600)` with `.verify(token) -> Principal`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_auth.py  (append)
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
    token = mint_token(tenant_id="acme", secret=HS_SECRET, ttl_seconds=-10)
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


def test_algorithm_confusion_hs_forged_with_rsa_pubkey_rejected():
    """Classic attack: forge an HS256 token using the RSA PUBLIC key as the HMAC
    secret. A verifier pinned to RS256 must reject it."""
    priv, pub = _rsa_keypair()
    kid = "key-1"
    jwks = _jwks_for(pub, kid)
    forged = _jwt.encode({"tenant_id": "attacker", "iat": int(time.time()),
                          "nbf": int(time.time()), "exp": int(time.time()) + 60},
                         pub, algorithm="HS256", headers={"kid": kid})
    v = JWTVerifier(alg="RS256", jwks_fetcher=lambda url: jwks)
    with pytest.raises(AuthError):
        v.verify(forged)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_auth.py -k "verify_ or algorithm_confusion" -v`
Expected: FAIL — `JWTVerifier` does not exist.

- [ ] **Step 3: Implement the verifier**

Create `providers/auth/jwt_verifier.py`:

```python
"""JWT verification. Algorithm is PINNED from config — never read from the token's
`alg` header — which defeats algorithm-confusion and `alg:none` attacks."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import jwt  # pyjwt

from core.interfaces import AuthError
from core.types import Principal


def _default_jwks_fetch(url: str) -> dict:
    import httpx

    resp = httpx.get(url, timeout=5.0)
    resp.raise_for_status()
    return resp.json()


class _JWKSCache:
    """Caches RSA public keys by `kid` with a TTL; refreshes once on an unknown kid."""

    def __init__(self, jwks_url: str, fetcher: Callable[[str], dict], ttl_seconds: int) -> None:
        self._url = jwks_url
        self._fetch = fetcher
        self._ttl = ttl_seconds
        self._keys: dict[str, object] = {}
        self._expires_at = 0.0

    def _refresh(self) -> None:
        jwks = self._fetch(self._url)
        self._keys = {
            k["kid"]: jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
            for k in jwks.get("keys", [])
            if "kid" in k
        }
        self._expires_at = time.time() + self._ttl

    def get(self, kid: str):
        if not self._keys or time.time() >= self._expires_at:
            self._refresh()
        if kid not in self._keys:
            self._refresh()  # unknown kid → single refresh (key rotation)
        key = self._keys.get(kid)
        if key is None:
            raise AuthError("unknown signing key", status=401)
        return key


class JWTVerifier:
    def __init__(
        self,
        *,
        alg: str,
        hs_secret: str = "",
        jwks_url: str = "",
        issuer: str = "",
        audience: str = "",
        leeway_seconds: int = 60,
        max_acl_tags: int = 32,
        jwks_fetcher: Callable[[str], dict] | None = None,
        jwks_ttl_seconds: int = 3600,
    ) -> None:
        self._alg = alg
        self._hs_secret = hs_secret
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway_seconds
        self._max_acl_tags = max_acl_tags
        self._jwks = (
            _JWKSCache(jwks_url, jwks_fetcher or _default_jwks_fetch, jwks_ttl_seconds)
            if alg == "RS256"
            else None
        )

    def _key_for(self, token: str):
        if self._alg == "HS256":
            return self._hs_secret
        # RS256: select the public key by the token's kid.
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as e:
            raise AuthError("malformed token header", status=401) from e
        kid = header.get("kid")
        if not kid:
            raise AuthError("token missing kid", status=401)
        try:
            return self._jwks.get(kid)  # type: ignore[union-attr]
        except AuthError:
            raise
        except Exception as e:  # JWKS fetch/parse failure → fail closed
            raise AuthError("jwks resolution failed", status=401) from e

    def verify(self, token: str) -> Principal:
        key = self._key_for(token)
        options = {"require": ["exp", "iat", "nbf"], "verify_aud": bool(self._audience)}
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[self._alg],  # PINNED — never the token's own alg
                issuer=self._issuer or None,
                audience=self._audience or None,
                leeway=self._leeway,
                options=options,
            )
        except jwt.ExpiredSignatureError as e:
            raise AuthError("token expired", status=401) from e
        except jwt.PyJWTError as e:
            raise AuthError("invalid token", status=401) from e

        tenant_id = claims.get("tenant_id")
        if not tenant_id or not str(tenant_id).strip():
            raise AuthError("missing tenant_id claim", status=403)
        acl_tags = tuple(claims.get("acl_tags") or ())
        if len(acl_tags) > self._max_acl_tags:
            raise AuthError("token presents too many acl_tags", status=403)
        return Principal(
            tenant_id=str(tenant_id),
            acl_tags=acl_tags,
            subject=str(claims.get("sub", "")),
            claims=dict(claims),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_auth.py -k "verify_ or algorithm_confusion" -v`
Expected: PASS (9 tests, including the algorithm-confusion rejection).

- [ ] **Step 5: Commit**

```bash
git add providers/auth/jwt_verifier.py tests/test_auth.py
git commit -m "Add algorithm-pinned JWTVerifier with HS256 and RS256/JWKS"
```

---

### Task 5: TenantAllowlist implementations

**Files:**
- Create: `providers/auth/allowlist.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces:
  - `providers.auth.allowlist.NullAllowlist()` — `allowed()` returns `None`.
  - `providers.auth.allowlist.StaticAllowlist(mapping: dict[str, list[str]])` + classmethod `from_file(path: str)`.
  - `providers.auth.allowlist.apply_allowlist(allowlist, tenant_id: str, acl_tags: tuple[str,...]) -> tuple[str,...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k allowlist -v`
Expected: FAIL — `providers.auth.allowlist` does not exist.

- [ ] **Step 3: Implement the allowlist**

Create `providers/auth/allowlist.py`:

```python
"""Per-tenant acl_tag allowlists. Default is NullAllowlist (claims pass through);
StaticAllowlist intersects claimed tags against a per-tenant grant (least privilege)."""

from __future__ import annotations

import json
from pathlib import Path


class NullAllowlist:
    def allowed(self, tenant_id: str) -> frozenset[str] | None:
        return None


class StaticAllowlist:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._map = {t: frozenset(tags) for t, tags in mapping.items()}

    @classmethod
    def from_file(cls, path: str) -> "StaticAllowlist":
        data = json.loads(Path(path).read_text())
        return cls(data)

    def allowed(self, tenant_id: str) -> frozenset[str] | None:
        # Unknown tenant → no permitted tags (fail closed / least privilege).
        return self._map.get(tenant_id, frozenset())


def apply_allowlist(allowlist, tenant_id: str, acl_tags: tuple[str, ...]) -> tuple[str, ...]:
    permitted = allowlist.allowed(tenant_id)
    if permitted is None:
        return tuple(acl_tags)
    return tuple(t for t in acl_tags if t in permitted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k allowlist -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add providers/auth/allowlist.py tests/test_auth.py
git commit -m "Add NullAllowlist and StaticAllowlist tenant tag resolution"
```

---

### Task 6: Registry builders

**Files:**
- Modify: `core/registry.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `JWTVerifier` (Task 4), allowlist impls (Task 5).
- Produces: `core.registry.build_auth_verifier(settings=None) -> JWTVerifier`; `core.registry.build_allowlist(settings=None) -> NullAllowlist | StaticAllowlist`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py  (append)
from core.config import Settings
from core.registry import build_auth_verifier, build_allowlist
from providers.auth.allowlist import NullAllowlist, StaticAllowlist


def test_build_auth_verifier_hs256():
    v = build_auth_verifier(Settings(jwt_alg="HS256", jwt_secret="s"))
    token = mint_token(tenant_id="acme", secret="s")
    assert v.verify(token).tenant_id == "acme"


def test_build_allowlist_null_by_default():
    assert isinstance(build_allowlist(Settings()), NullAllowlist)


def test_build_allowlist_static_from_source(tmp_path):
    p = tmp_path / "acl.json"
    p.write_text(json.dumps({"acme": ["finance"]}))
    al = build_allowlist(Settings(acl_allowlist_source=str(p)))
    assert isinstance(al, StaticAllowlist)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k build_ -v`
Expected: FAIL — `build_auth_verifier`/`build_allowlist` not defined.

- [ ] **Step 3: Implement the builders**

Append to `core/registry.py`:

```python
def build_auth_verifier(settings: Settings | None = None):
    """Build the JWT verifier from config (the only place its class is named)."""
    s = settings or get_settings()
    from providers.auth.jwt_verifier import JWTVerifier

    return JWTVerifier(
        alg=s.jwt_alg,
        hs_secret=s.jwt_secret,
        jwks_url=s.jwks_url,
        issuer=s.jwt_issuer,
        audience=s.jwt_audience,
        leeway_seconds=s.jwt_leeway_seconds,
        max_acl_tags=s.max_acl_tags,
    )


def build_allowlist(settings: Settings | None = None):
    """Build the tenant allowlist: NullAllowlist unless acl_allowlist_source is set."""
    s = settings or get_settings()
    if not s.acl_allowlist_source:
        from providers.auth.allowlist import NullAllowlist

        return NullAllowlist()
    from providers.auth.allowlist import StaticAllowlist

    return StaticAllowlist.from_file(s.acl_allowlist_source)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k build_ -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/registry.py tests/test_auth.py
git commit -m "Wire auth verifier and allowlist builders into the registry"
```

---

### Task 7: API — require_principal dependency, verified /query, inverted spoof tests

**Files:**
- Create: `app/auth.py`
- Modify: `app/api.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `build_auth_verifier`/`build_allowlist` (Task 6), `apply_allowlist` (Task 5), `AuthError` (Task 2), `Principal` (Task 2), `mint_token` (Task 3).
- Produces: `app.auth.get_verifier()`, `app.auth.get_allowlist()`, `app.auth.require_principal(...) -> Principal` (all FastAPI dependencies).

- [ ] **Step 1: Write the failing tests (rewrite the identity tests)**

Replace the tenant-plumbing tests in `tests/test_app.py`. Remove `test_tenant_flows_from_request_body`, `test_tenant_header_overrides_body`, and `test_acl_tags_flow_from_request`. Update the `client` fixture and happy-path test to authenticate, and add the new auth tests. Concretely:

Add near the top of `tests/test_app.py`:

```python
from app.auth import get_verifier
from providers.auth.jwt_verifier import JWTVerifier
from providers.auth.dev_signer import mint_token

TEST_SECRET = "app-test-secret"


def _auth_header(tenant_id="tenant_a", acl_tags=()):
    token = mint_token(tenant_id=tenant_id, acl_tags=list(acl_tags), secret=TEST_SECRET)
    return {"Authorization": f"Bearer {token}"}
```

Update the `client` fixture to also override the verifier with a known secret:

```python
@pytest.fixture()
def client(fake_pipeline):
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_verifier] = lambda: JWTVerifier(
        alg="HS256", hs_secret=TEST_SECRET, max_acl_tags=32
    )
    yield TestClient(app)
    app.dependency_overrides.clear()
```

Update the happy-path test to send a token, and add the new tests:

```python
def test_query_returns_200_with_answer_and_citations(client, fake_pipeline):
    resp = client.post("/query", json={"question": "What is the capital of France?"},
                       headers=_auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == CANNED_ANSWER
    assert data["refused"] is False


def test_query_without_token_is_401(client):
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_query_bad_signature_is_401(client):
    bad = mint_token(tenant_id="tenant_a", secret="not-the-secret")
    resp = client.post("/query", json={"question": "hi"},
                       headers={"Authorization": f"Bearer {bad}"})
    assert resp.status_code == 401


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


def test_oversized_question_is_422(client, monkeypatch):
    from core import config
    config.get_settings.cache_clear()
    monkeypatch.setenv("MAX_QUESTION_CHARS", "10")
    config.get_settings.cache_clear()
    resp = client.post("/query", json={"question": "x" * 50}, headers=_auth_header())
    assert resp.status_code == 422
    config.get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL — `app.auth` does not exist / `get_verifier` unimportable.

- [ ] **Step 3: Implement app/auth.py**

Create `app/auth.py`:

```python
"""FastAPI auth dependencies: verify a Bearer token into a Principal.

Identity (tenant_id, acl_tags) comes ONLY from the verified token — there is no
client-controlled header/body identity path. get_verifier/get_allowlist are
FastAPI dependencies so tests can override them with a known-secret verifier.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from core.config import get_settings
from core.interfaces import AuthError
from core.registry import build_allowlist, build_auth_verifier
from core.types import Principal
from providers.auth.allowlist import apply_allowlist

_verifier = None
_allowlist = None


def get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = build_auth_verifier()
    return _verifier


def get_allowlist():
    global _allowlist
    if _allowlist is None:
        _allowlist = build_allowlist()
    return _allowlist


def require_principal(
    authorization: str | None = Header(default=None),
    verifier=Depends(get_verifier),
    allowlist=Depends(get_allowlist),
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    try:
        principal = verifier.verify(token)
    except AuthError as e:
        headers = {"WWW-Authenticate": "Bearer"} if e.status == 401 else None
        raise HTTPException(status_code=e.status, detail=e.detail, headers=headers) from e

    tags = apply_allowlist(allowlist, principal.tenant_id, principal.acl_tags)
    if tags != principal.acl_tags:
        principal = principal.model_copy(update={"acl_tags": tags})
    return principal


def demo_principal(tenant_id: str, *, acl_tags=()) -> Principal:
    """Mint + verify a token for the Streamlit demo, exercising the real auth path
    so the demo proves isolation rather than trusting a dropdown. Requires the dev
    signer (HS256 secret + flag); raises otherwise."""
    from providers.auth.dev_signer import mint_token

    s = get_settings()
    if not s.auth_dev_signer_enabled or not s.jwt_secret:
        raise RuntimeError("demo token minting requires auth_dev_signer_enabled and jwt_secret")
    token = mint_token(tenant_id=tenant_id, acl_tags=list(acl_tags), secret=s.jwt_secret,
                       issuer=s.jwt_issuer, audience=s.jwt_audience)
    return build_auth_verifier(s).verify(token)
```

- [ ] **Step 4: Implement app/api.py changes**

Replace the imports, `QueryRequest`, and `query` handler in `app/api.py`. The new file body (from the module docstring down) is:

```python
"""FastAPI application for the Production RAG system.

Security: tenant identity is derived ONLY from a cryptographically verified JWT
(see app.auth.require_principal). There is no client-controlled identity path —
the request body carries only the question.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.auth import require_principal
from core.config import get_settings
from core.types import Principal

_pipeline: Any = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from core.pipeline import build

        _pipeline = build(version="full", dataset=None)
    return _pipeline


app = FastAPI(title="Production RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    # Identity (tenant_id/acl_tags) intentionally REMOVED — it comes only from the
    # verified token. The body carries only the question.
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_ids: list[str]
    usage: dict
    refused: bool


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    principal: Principal = Depends(require_principal),
    pipeline=Depends(get_pipeline),
):
    if len(body.question) > get_settings().max_question_chars:
        raise HTTPException(status_code=422, detail="question too long")

    acl = principal.to_acl()  # identity from the verified token only
    result = pipeline.run(body.question, acl)

    return QueryResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        retrieved_ids=result.get("retrieved_ids", []),
        usage=result.get("usage", {}),
        refused=result.get("refused", False),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: PASS — happy path with token, 401 without token, 401 bad signature, identity-from-token-only (spoof ignored), 422 oversized.

- [ ] **Step 6: Commit**

```bash
git add app/auth.py app/api.py tests/test_app.py
git commit -m "Require verified JWT on /query; derive ACL only from the token"
```

---

### Task 8: Streamlit demo through the real auth path

**Files:**
- Modify: `app/demo.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `demo_principal` (Task 7).
- Produces: demo builds its `ACLContext` from a minted+verified token instead of directly from the dropdown.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -k demo_principal_roundtrip -v`
Expected: PASS already for the helper (added in Task 7) — but the demo module itself must use it. If Task 7's `demo_principal` is present this passes; the remaining work is wiring `app/demo.py`. If it fails, implement per Step 3.

- [ ] **Step 3: Wire the demo to the auth path**

In `app/demo.py`, inside `_run_app()`, replace the ACL construction. Change:

```python
        acl = ACLContext(tenant_id=tenant)
```

to:

```python
        from app.auth import demo_principal  # noqa: PLC0415

        try:
            acl = demo_principal(tenant).to_acl()
        except RuntimeError:
            st.error("Demo auth is disabled. Set AUTH_DEV_SIGNER_ENABLED=true and JWT_SECRET to run the demo.")
            st.stop()
```

The `from core.types import ACLContext` import in `_run_app` may now be unused; remove it if so (keep the file importing cleanly).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -k "demo" -v`
Expected: PASS (`test_demo_imports_cleanly` and `test_demo_principal_roundtrip`).

- [ ] **Step 5: Commit**

```bash
git add app/demo.py tests/test_app.py
git commit -m "Route the Streamlit demo through the real JWT auth path"
```

---

### Task 9: mint_token CLI + env docs

**Files:**
- Create: `scripts/mint_token.py`
- Modify: `infra/.env.example`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `mint_token` (Task 3), `get_settings` (Task 1).
- Produces: `python -m scripts.mint_token --tenant <id> [--tags a,b] [--ttl N]` prints a Bearer token; `scripts.mint_token.build_token(settings, tenant, tags, ttl) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py  (append)
def test_cli_build_token_verifies():
    from core.config import Settings
    from scripts.mint_token import build_token

    s = Settings(auth_dev_signer_enabled=True, jwt_secret="cli-secret")
    token = build_token(s, tenant="acme", tags=["finance"], ttl=60)
    v = build_auth_verifier(s)
    assert v.verify(token).tenant_id == "acme"


def test_cli_build_token_requires_dev_signer():
    from core.config import Settings
    from scripts.mint_token import build_token

    s = Settings(auth_dev_signer_enabled=False, jwt_secret="")
    with pytest.raises(RuntimeError):
        build_token(s, tenant="acme", tags=[], ttl=60)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -k cli_build_token -v`
Expected: FAIL — `scripts.mint_token` does not exist.

- [ ] **Step 3: Implement the CLI**

Create `scripts/mint_token.py`:

```python
"""Mint a dev JWT for local testing / the demo.

    uv run python -m scripts.mint_token --tenant acme --tags finance,hr --ttl 3600

Requires auth_dev_signer_enabled=true and a jwt_secret (HS256). A prod instance
(RS256, public-key-only) cannot mint and this will refuse.
"""

from __future__ import annotations

import argparse

from core.config import Settings, get_settings
from providers.auth.dev_signer import mint_token


def build_token(settings: Settings, *, tenant: str, tags: list[str], ttl: int) -> str:
    if not settings.auth_dev_signer_enabled or not settings.jwt_secret:
        raise RuntimeError(
            "minting requires auth_dev_signer_enabled=true and a jwt_secret (HS256)"
        )
    return mint_token(
        tenant_id=tenant,
        acl_tags=tags,
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        ttl_seconds=ttl,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a dev JWT.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--tags", default="", help="comma-separated acl_tags")
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    token = build_token(get_settings(), tenant=args.tenant, tags=tags, ttl=args.ttl)
    print(token)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Document the knobs in infra/.env.example**

Append to `infra/.env.example`:

```bash
# --- Auth & tenancy (SP1) ---
# Dev defaults: HS256 with a local secret + dev signer on. NEVER use these in prod.
APP_ENV=dev
AUTH_ENABLED=true
JWT_ALG=HS256
JWT_SECRET=change-me-dev-secret
# JWKS_URL=            # RS256/prod only
JWT_ISSUER=
JWT_AUDIENCE=
JWT_LEEWAY_SECONDS=60
# ACL_ALLOWLIST_SOURCE=config/acl_allowlist.json   # optional per-tenant tag grants
AUTH_DEV_SIGNER_ENABLED=true
MAX_QUESTION_CHARS=8000
MAX_ACL_TAGS=32
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -k cli_build_token -v`
Expected: PASS (2 tests).

Then verify the CLI end-to-end:
Run: `AUTH_DEV_SIGNER_ENABLED=true JWT_SECRET=demo uv run python -m scripts.mint_token --tenant acme --tags finance`
Expected: prints a `eyJ...` token string.

- [ ] **Step 6: Commit**

```bash
git add scripts/mint_token.py infra/.env.example tests/test_auth.py
git commit -m "Add mint_token CLI and document auth env knobs"
```

---

### Task 10: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new `tests/test_auth.py` and rewritten `tests/test_app.py`. No test still asserts header/body-derived identity.

- [ ] **Step 2: Lint**

Run: `uv run ruff check core providers app scripts tests`
Expected: clean (fix any unused imports, e.g. a leftover `ACLContext` import in `app/demo.py` or `app/api.py`).

- [ ] **Step 3: Grep for regressions in the identity path**

Run: `grep -rn "X-Tenant-Id\|x_tenant_id" app/ tests/`
Expected: only the `test_identity_comes_only_from_verified_token` reference (the attacker-attempt header that must be ignored). No production code reads it.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "SP1: lint cleanup and full-suite verification"
```

---

## Self-Review (completed during authoring)

**Spec coverage:** §5.1 Principal → T2. §5.2 AuthVerifier/AuthError → T2. §5.3 JWTVerifier (alg-pin, HS/RS, JWKS) → T4. §5.4 TenantAllowlist → T5. §5.5 registry builders → T6. §5.6 require_principal → T7. §5.7 dev signer + CLI → T3/T9. §6 hardening: alg-pin → T4 (+ confusion test), aud/iss/nbf/iat → T4, dev-signer-impossible-in-prod → T1 (boot check) + T3 (secret-gated) + T7 (`demo_principal` guard), revocation documented → spec §6/§13 (no code), JWKS cache → T4, no secret logging → constraint. §7 data flow → T7. §8 API changes (body identity removed, error shapes, WWW-Authenticate) → T7. §9 config → T1. §10 demo → T8. §11 testing → T1–T9. §12 files → all. Every spec section maps to a task.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every run step shows the command + expected result.

**Type consistency:** `Principal(tenant_id, acl_tags, subject, claims)` and `.to_acl()` used identically in T2/T4/T7/T8. `AuthError(detail, status)` consistent T2/T4/T7. `mint_token(*, tenant_id, acl_tags, secret, issuer, audience, ttl_seconds, ...)` consistent T3/T4/T7/T9. `JWTVerifier(alg=, hs_secret=, ...)` consistent T4/T6/T7. `apply_allowlist(allowlist, tenant_id, acl_tags)` consistent T5/T7. `build_auth_verifier`/`build_allowlist` consistent T6/T7/T9.
