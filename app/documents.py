"""Documents API: async upload + tenant-scoped status.

Upload is fire-and-forget: the request validates the content type and size,
persists the raw bytes to the blob store, registers a `processing` row, and
enqueues the ingest job. Parsing/chunking/embedding happen out of band in the
arq worker (see ingest.worker), so uploads return 202 immediately.

Security: tenant identity comes ONLY from the verified token (require_principal).
The blob key hashes the tenant segment (matching the manifest/sparse stores) so a
hostile tenant id can never escape the blob-store root. Status reads are tenant
scoped in the registry, so one tenant cannot observe another's documents.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import require_principal
from core.types import DocumentRecord, DocumentStatus, Principal
from ingest.parsers.base import ParserError, ParserRegistry

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Dependency providers — cached singletons, overridable in tests. Built lazily
# so importing this module stays cheap (no store/pool construction at import).
# ---------------------------------------------------------------------------

_registry = None
_blobs = None
_parsers = None
_enqueuer: Callable[[str], Awaitable[None]] | None = None


def get_registry():
    global _registry
    if _registry is None:
        from core.registry import build_document_registry

        _registry = build_document_registry()
    return _registry


def get_blobs():
    global _blobs
    if _blobs is None:
        from core.registry import build_blob_store

        _blobs = build_blob_store()
    return _blobs


def get_parsers() -> ParserRegistry:
    global _parsers
    if _parsers is None:
        from core.registry import build_parser_registry

        _parsers = build_parser_registry()
    return _parsers


def get_enqueuer() -> Callable[[str], Awaitable[None]]:
    global _enqueuer
    if _enqueuer is None:
        _enqueuer = _build_arq_enqueuer()
    return _enqueuer


_pool = None


def _build_arq_enqueuer() -> Callable[[str], Awaitable[None]]:
    """Default enqueuer: submit `ingest_document` to arq/Redis. The pool is
    created on first use and reused (FastAPI runs on a single event loop)."""

    async def enqueue(document_id: str) -> None:
        global _pool
        if _pool is None:
            from arq import create_pool

            from ingest.worker import WorkerSettings

            _pool = await create_pool(WorkerSettings.redis_settings())
        await _pool.enqueue_job("ingest_document", document_id)

    return enqueue


def _blob_key(tenant_id: str, document_id: str) -> str:
    """Namespace blobs by a hashed tenant segment so an adversarial tenant id
    cannot traverse outside the blob-store root (matches manifest/sparse stores)."""
    tenant_safe = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return f"{tenant_safe}/{document_id}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
    blobs=Depends(get_blobs),
    parsers: ParserRegistry = Depends(get_parsers),
    enqueue: Callable[[str], Awaitable[None]] = Depends(get_enqueuer),
):
    content_type = file.content_type or "application/octet-stream"

    # Reject unsupported types before reading the body (415).
    try:
        parsers.resolve(content_type)
    except ParserError:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {content_type}")

    raw = await file.read()

    # Enforce the size limit after reading (413). UploadFile streams to a spooled
    # temp file, so this does not pin arbitrarily large bodies in memory.
    try:
        parsers.guard_size(raw)
    except ParserError:
        raise HTTPException(status_code=413, detail="upload exceeds maximum size")

    document_id = uuid.uuid4().hex
    blob_key = _blob_key(principal.tenant_id, document_id)
    blobs.put(blob_key, raw)

    registry.create(
        DocumentRecord(
            document_id=document_id,
            tenant_id=principal.tenant_id,
            filename=file.filename or "upload",
            content_type=content_type,
            size_bytes=len(raw),
            status=DocumentStatus.PROCESSING,
            blob_key=blob_key,
        )
    )
    await enqueue(document_id)

    return {"document_id": document_id, "status": DocumentStatus.PROCESSING.value}


@router.get("/{document_id}")
def get_document(
    document_id: str,
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
):
    record = registry.get(document_id, principal.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {
        "document_id": record.document_id,
        "filename": record.filename,
        "content_type": record.content_type,
        "size_bytes": record.size_bytes,
        "status": record.status.value,
        "chunk_count": record.chunk_count,
        "error": record.error,
    }
