from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config import Settings, get_settings
from core.types import ACLContext, DocumentStatus
from ingest.chunking import chunk_document

logger = logging.getLogger(__name__)


@dataclass
class IngestDeps:
    registry: object   # DocumentRegistry
    blobs: object      # BlobStore
    parsers: object    # ParserRegistry
    ingestor: object   # IncrementalIngestor
    settings: Settings


def _pii_process(docs, settings, tenant_id):
    """Reuse the ingest PII policy. redact => clean doc text; keep => tag later."""
    from core.registry import build_pii_detector
    from ingest.audit import PIIAuditLog
    from ingest.run import _apply_pii_ingest_policy

    detector = build_pii_detector(settings)
    audit = PIIAuditLog(settings)
    clean_docs, _ = _apply_pii_ingest_policy(docs, settings, detector, audit)
    return clean_docs, detector, audit


def run_ingest(deps: IngestDeps, document_id: str) -> None:
    """Pure worker body: parse → PII → chunk → incremental ingest. Fail-closed:
    any error marks the document `failed` and leaves no partial 'ready'."""
    rec = deps.registry.get_privileged(document_id)
    if rec is None:
        logger.error("ingest: document %s not found", document_id)
        return

    tenant_id = rec.tenant_id
    acl = ACLContext(tenant_id=tenant_id, acl_tags=())
    try:
        parser = deps.parsers.resolve(rec.content_type)
        raw = deps.blobs.get(rec.blob_key)
        docs = parser.parse(raw, rec.filename, rec.content_type,
                            doc_id=document_id, tenant_id=tenant_id, acl_tags=())
        if rec.collection_id:
            docs = [d.model_copy(update={"collection_id": rec.collection_id}) for d in docs]
        clean_docs, detector, audit = _pii_process(docs, deps.settings, tenant_id)

        chunks = []
        for doc in clean_docs:
            chunks.extend(chunk_document(doc))
        if deps.settings.pii_mode == "keep":
            for ch in chunks:
                spans = detector.detect(ch.text)
                if spans:
                    ch.metadata["pii_types"] = sorted({s.type for s in spans})
                    audit.record(tenant_id=ch.tenant_id, doc_id=ch.doc_id,
                                 chunk_id=ch.chunk_id, text=ch.text, spans=spans)

        n = deps.ingestor.ingest_document(tenant_id, document_id, chunks, acl)
        deps.registry.set_status(document_id, tenant_id, DocumentStatus.READY, chunk_count=n)
    except Exception as e:  # fail-closed
        logger.exception("ingest failed for %s", document_id)
        deps.registry.set_status(document_id, tenant_id, DocumentStatus.FAILED,
                                 error=type(e).__name__)


def run_delete(deps: IngestDeps, document_id: str) -> None:
    """Pure delete body: purge chunks (dense+sparse) + manifest + blob, then the
    registry row LAST. Fail-closed: an error marks the doc `failed`, never a
    half-deleted ghost. Every underlying delete is idempotent, so retry is safe."""
    rec = deps.registry.get_privileged(document_id)
    if rec is None:
        logger.info("delete: document %s already gone", document_id)
        return
    tenant_id = rec.tenant_id
    acl = ACLContext(tenant_id=tenant_id, acl_tags=())
    try:
        deps.ingestor.delete_document(tenant_id, document_id, acl)
        deps.blobs.delete(rec.blob_key)
        deps.registry.delete(document_id, tenant_id)
    except Exception as e:  # fail-closed
        logger.exception("delete failed for %s", document_id)
        deps.registry.set_status(document_id, tenant_id, DocumentStatus.FAILED,
                                 error=type(e).__name__)


def _build_deps(settings: Settings) -> IngestDeps:
    from core.registry import (build_blob_store, build_document_registry,
                               build_incremental_ingestor, build_parser_registry)
    return IngestDeps(
        registry=build_document_registry(settings),
        blobs=build_blob_store(settings),
        parsers=build_parser_registry(settings),
        ingestor=build_incremental_ingestor(settings),
        settings=settings,
    )


async def ingest_document(ctx, document_id: str) -> None:
    """arq task entrypoint. Deps are built once per worker and cached on ctx."""
    deps = ctx.get("deps")
    if deps is None:
        from core.config import get_settings
        deps = _build_deps(get_settings())
        ctx["deps"] = deps
    run_ingest(deps, document_id)


async def delete_document(ctx, document_id: str) -> None:
    """arq task entrypoint for async document deletion."""
    deps = ctx.get("deps")
    if deps is None:
        from core.config import get_settings
        deps = _build_deps(get_settings())
        ctx["deps"] = deps
    run_delete(deps, document_id)


try:
    from arq.connections import RedisSettings as _RedisSettings
except ModuleNotFoundError:  # arq ships in the 'app' extra; the base test env omits it
    _RedisSettings = None


class WorkerSettings:
    """arq worker configuration. Run it with:

        arq ingest.worker.WorkerSettings

    `redis_settings` MUST be a concrete ``RedisSettings`` in the class body: arq's
    ``get_kwargs`` copies attributes straight out of the class ``__dict__`` into the
    ``Worker`` (it does not call them), so a method/property is passed through
    uninterpreted and pool creation fails. It is only materialised when arq is
    importable — i.e. the worker image — which keeps ``import ingest.worker``
    dependency-free for unit tests that drive ``run_ingest`` directly.
    """

    functions = [ingest_document, delete_document]

    if _RedisSettings is not None:
        redis_settings = _RedisSettings.from_dsn(get_settings().redis_url)
