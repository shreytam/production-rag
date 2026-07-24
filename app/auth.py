"""FastAPI auth dependencies: verify a Bearer token into a Principal.

Identity (tenant_id, acl_tags) comes ONLY from the verified token — there is no
client-controlled header/body identity path. get_verifier/get_allowlist are
FastAPI dependencies so tests can override them with a known-secret verifier.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

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
