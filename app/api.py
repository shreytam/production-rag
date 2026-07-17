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

        _pipeline = build(version="full", corpus=None)
    return _pipeline


# Guardrails are enforced inside the shared RAGPipeline (input + output guards,
# config-gated via guardrails_enabled). A blocked request comes back as a normal
# response with refused=True, so the API just surfaces that flag.

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Production RAG API", version="1.0.0")

# Async document ingestion API (upload + tenant-scoped status).
from app.documents import router as documents_router  # noqa: E402

app.include_router(documents_router)


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
