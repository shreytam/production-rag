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
