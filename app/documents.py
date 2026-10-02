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
import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import Query as QueryParam
from starlette.concurrency import run_in_threadpool

from app.auth import require_principal
from app.ratelimit import upload_rate_limit
from core.types import DocumentRecord, DocumentStatus, Principal
from ingest.parsers.base import ParserError, ParserRegistry

router = APIRouter(prefix="/documents", tags=["documents"])

log = logging.getLogger(__name__)

_MAX_COLLECTION_ID = 128
_READ_CHUNK = 1024 * 1024  # 1 MiB


def _validate_collection_id(value: str) -> str:
    """Reject collection ids that are too long or contain control characters."""
    if len(value) > _MAX_COLLECTION_ID or any(ord(ch) < 32 for ch in value):
        raise HTTPException(status_code=422, detail="invalid collection_id")
    return value


# ---------------------------------------------------------------------------
# Dependency providers — cached singletons, overridable in tests. Built lazily
# so importing this module stays cheap (no store/pool construction at import).
# ---------------------------------------------------------------------------

_registry = None
_blobs = None
_parsers = None
_enqueuer: Callable[..., Awaitable[None]] | None = None


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


def get_enqueuer() -> Callable[..., Awaitable[None]]:
    global _enqueuer
    if _enqueuer is None:
        _enqueuer = _build_arq_enqueuer()
    return _enqueuer


_pool = None

_ACTION_TO_FN = {"ingest": "ingest_document", "delete": "delete_document"}


def _build_arq_enqueuer() -> Callable[..., Awaitable[None]]:
    """Default enqueuer: submit an ingest/delete job to arq/Redis. The pool is
    created on first use and reused (FastAPI runs on a single event loop)."""

    async def enqueue(document_id: str, action: str = "ingest") -> None:
        global _pool
        if _pool is None:
            from arq import create_pool

            from ingest.worker import WorkerSettings

            # Attribute, not a factory — arq reads it straight out of the class
            # __dict__ and never calls it (see ingest.worker.WorkerSettings).
            _pool = await create_pool(WorkerSettings.redis_settings)
        await _pool.enqueue_job(_ACTION_TO_FN[action], document_id)

    return enqueue


def _blob_key(tenant_id: str, document_id: str) -> str:
    """Namespace blobs by a hashed tenant segment so an adversarial tenant id
    cannot traverse outside the blob-store root (matches manifest/sparse stores)."""
    tenant_safe = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return f"{tenant_safe}/{document_id}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202, dependencies=[Depends(upload_rate_limit)])
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str = Form(""),
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
    blobs=Depends(get_blobs),
    parsers: ParserRegistry = Depends(get_parsers),
    enqueue: Callable[..., Awaitable[None]] = Depends(get_enqueuer),
):
    collection_id = _validate_collection_id(collection_id)

    content_type = file.content_type or "application/octet-stream"

    # Reject unsupported types before reading the body (415).
    try:
        parsers.resolve(content_type)
    except ParserError:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {content_type}")

    # Read in bounded chunks and abort the moment the cap is crossed, so a huge
    # body is never held in memory (Starlette already spooled it to a temp file).
    limit = parsers.max_bytes
    buf = bytearray()
    while chunk := await file.read(_READ_CHUNK):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail="upload exceeds maximum size")
    raw = bytes(buf)

    # The client-declared Content-Type is untrusted: verify the bytes match it.
    try:
        parsers.validate_content(content_type, raw)
    except ParserError:
        raise HTTPException(status_code=415, detail="content does not match declared content type")

    document_id = uuid.uuid4().hex
    blob_key = _blob_key(principal.tenant_id, document_id)
    # Registry/blob I/O is synchronous (disk, psycopg): keep it off the event loop.
    await run_in_threadpool(blobs.put, blob_key, raw)

    await run_in_threadpool(
        registry.create,
        DocumentRecord(
            document_id=document_id,
            tenant_id=principal.tenant_id,
            filename=file.filename or "upload",
            content_type=content_type,
            size_bytes=len(raw),
            status=DocumentStatus.PROCESSING,
            blob_key=blob_key,
            collection_id=collection_id,
        ),
    )
    try:
        await enqueue(document_id)
    except Exception:
        # Never strand a PROCESSING row / orphan blob nobody will ever process.
        log.exception("ingest enqueue failed for %s", document_id)
        try:
            await run_in_threadpool(registry.delete, document_id, principal.tenant_id)
            await run_in_threadpool(blobs.delete, blob_key)
        except Exception:
            log.exception("cleanup after enqueue failure failed for %s", document_id)
        raise HTTPException(status_code=503, detail="ingest queue unavailable, retry later")

    return {"document_id": document_id, "status": DocumentStatus.PROCESSING.value}


@router.get("")
def list_documents(
    collection_id: str | None = QueryParam(default=None),
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
):
    records = registry.list(principal.tenant_id)
    if collection_id is not None:
        records = [r for r in records if r.collection_id == collection_id]
    return [
        {
            "document_id": r.document_id,
            "filename": r.filename,
            "content_type": r.content_type,
            "size_bytes": r.size_bytes,
            "status": r.status.value,
            "chunk_count": r.chunk_count,
            "collection_id": r.collection_id,
            "error": r.error,
        }
        for r in records
    ]


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
        "collection_id": record.collection_id,
        "error": record.error,
    }


@router.delete("/{document_id}", status_code=202)
async def delete_document(
    document_id: str,
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
    enqueue: Callable[..., Awaitable[None]] = Depends(get_enqueuer),
):
    record = await run_in_threadpool(registry.get, document_id, principal.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    if record.status == DocumentStatus.PROCESSING:
        # Deleting under a running ingest job would race it; client retries later.
        raise HTTPException(status_code=409, detail="document is still processing")
    await run_in_threadpool(
        registry.set_status, document_id, principal.tenant_id, DocumentStatus.DELETING
    )
    try:
        await enqueue(document_id, "delete")
    except Exception:
        log.exception("delete enqueue failed for %s", document_id)
        try:
            await run_in_threadpool(
                registry.set_status, document_id, principal.tenant_id, record.status,
                error=record.error, chunk_count=record.chunk_count,
            )
        except Exception:
            log.exception("status revert failed for %s", document_id)
        raise HTTPException(status_code=503, detail="ingest queue unavailable, retry later")
    return {"document_id": document_id, "status": DocumentStatus.DELETING.value}
