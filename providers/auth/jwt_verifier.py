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
        options = {"require": ["exp"], "verify_aud": bool(self._audience)}
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
