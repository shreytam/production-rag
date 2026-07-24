"""Local test console — a single-page UI that drives the real HTTP API.

The console is a plain static page; every action it takes is an ordinary
authenticated call to `/documents` and `/query`, so exercising it exercises the
same path any other client takes (async ingest, the arq worker, JWT auth,
tenant scoping) rather than an in-process shortcut.

SECURITY: `/ui` and `/ui/token` mint and serve dev credentials, so both are
gated on `auth_dev_signer_enabled` *and* the presence of an HS256 secret. Both
return 404 otherwise, and `core/config.py` refuses to boot with the flag set
when `app_env=prod` — a production deploy therefore exposes neither route.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.config import get_settings

router = APIRouter(prefix="/ui", tags=["ui"])

_CONSOLE_HTML = Path(__file__).parent / "static" / "console.html"


@lru_cache(maxsize=1)
def _console_html() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _require_dev_signer():
    """Return settings, or 404 if this instance must not mint dev tokens.

    404 rather than 403 so a locked-down deploy does not advertise that a
    console exists at all.
    """
    s = get_settings()
    if not s.auth_dev_signer_enabled or not s.jwt_secret:
        raise HTTPException(status_code=404, detail="Not Found")
    return s


class TokenRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    acl_tags: list[str] = Field(default_factory=list, max_length=32)


@router.get("", response_class=HTMLResponse)
def console() -> HTMLResponse:
    _require_dev_signer()
    return HTMLResponse(_console_html())


@router.post("/token")
def mint_dev_token(body: TokenRequest) -> dict:
    """Mint a short-lived HS256 token for the console to send as a Bearer header.

    The console never sees the secret; it holds only the resulting token, and
    that token is verified by the same `require_principal` dependency as any
    other request.
    """
    s = _require_dev_signer()
    from providers.auth.dev_signer import mint_token  # noqa: PLC0415

    token = mint_token(
        tenant_id=body.tenant_id,
        acl_tags=body.acl_tags,
        secret=s.jwt_secret,
        issuer=s.jwt_issuer,
        audience=s.jwt_audience,
    )
    return {"token": token, "tenant_id": body.tenant_id, "acl_tags": body.acl_tags}
