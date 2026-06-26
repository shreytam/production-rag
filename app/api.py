"""FastAPI application for the Production RAG system.

Security note: tenant identity is derived exclusively from the authenticated
request (X-Tenant-Id header or request body), never from the question text.
In production, X-Tenant-Id would be a verified JWT claim extracted by an
auth middleware; the body fallback exists only for demo convenience and MUST
be removed in production.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header
from pydantic import BaseModel

from core.types import ACLContext

# ---------------------------------------------------------------------------
# Pipeline singleton — built lazily on first request so that importing this
# module is cheap (no network / model-loading at import time).
# ---------------------------------------------------------------------------

_pipeline: Any = None


def get_pipeline():
    """Return the cached RAGPipeline, building it on the first call."""
    global _pipeline
    if _pipeline is None:
        from core.pipeline import build  # heavy import deferred intentionally
        _pipeline = build(version="full", dataset=None)
    return _pipeline


# Guardrails are enforced inside the shared RAGPipeline (input + output guards,
# config-gated via guardrails_enabled). A blocked request comes back as a normal
# response with refused=True, so the API just surfaces that flag.

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Production RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    # In production, tenant_id is extracted from a verified token/session,
    # never trusted from the raw request body.  Here we accept it in the body
    # only as a demo fallback when the X-Tenant-Id header is absent.
    tenant_id: str = "public"
    acl_tags: list[str] = []


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
    pipeline=Depends(get_pipeline),
    x_tenant_id: str | None = Header(default=None),
):
    # Tenant identity: header takes precedence over body (closer to a real auth
    # flow where the header would carry a verified claim).
    tenant_id = x_tenant_id if x_tenant_id is not None else body.tenant_id

    # Build ACL server-side; question text plays no part.
    acl = ACLContext(tenant_id=tenant_id, acl_tags=tuple(body.acl_tags))

    # Guardrails (input + output) run inside pipeline.run; a blocked request
    # returns refused=True rather than raising.
    result = pipeline.run(body.question, acl)

    return QueryResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        retrieved_ids=result.get("retrieved_ids", []),
        usage=result.get("usage", {}),
        refused=result.get("refused", False),
    )
