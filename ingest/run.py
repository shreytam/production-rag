"""CLI entry point for PDF ingestion into the RAG stack.

Usage
-----
    python -m ingest.run --input PATH [--tenant public] [--acl-tag TAG ...]
                         [--collection ID] [--contextual] [--recursive]

``--input`` accepts a single PDF file or a directory (a ``*.pdf`` glob; add
``--recursive`` to descend into subdirectories). Every PDF gets a
deterministic doc_id — uuid5 over its resolved absolute path — so re-running
the CLI over an unchanged tree is cheap: IncrementalIngestor diffs each
document's chunks against its persisted manifest and skips everything that
did not change (no re-embed, no upsert).

Per document the pipeline mirrors ``ingest/worker.py``:
UnstructuredParser.parse → PII policy (redact/keep per settings) →
chunk_document → optional ContextualPrefixer (``--contextual``) →
IncrementalIngestor.ingest_document (embed → dense upsert → per-tenant
sparse index → JSONL manifest).

All components are built through ``core.registry`` factories. There is no
dataset/corpus loading, no tenant-splitting, and no legacy BM25-pickle or
golden-eval export here.

NOTE: This module makes real network calls (embedding API, vector store,
optional LLM for ``--contextual``). It is NOT exercised by the offline test
suite. All heavy imports happen inside main()/the helpers so importing the
module stays safe.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from core.config import Settings
from core.types import ACLContext, Chunk, Document
from core.interfaces import PIIDetector
from ingest.audit import PIIAuditLog

PDF_CONTENT_TYPE = "application/pdf"
PDF_SUFFIXES = {".pdf"}


def _doc_id_for(path: Path) -> str:
    """Deterministic document id: uuid5 over the resolved absolute path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(path).resolve())))


def _collect_pdf_paths(raw_input: str, recursive: bool) -> list[Path]:
    """Expand --input into a deterministic, sorted list of PDF paths."""
    resolved = Path(raw_input).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in PDF_SUFFIXES:
            raise SystemExit(f"[ingest] error: not a PDF file: {resolved}")
        return [resolved]
    if resolved.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdfs = sorted(p for p in resolved.glob(pattern) if p.is_file())
        if not pdfs:
            where = "recursively under" if recursive else "in"
            raise SystemExit(f"[ingest] error: no PDF files found {where} {resolved}")
        return pdfs
    raise SystemExit(f"[ingest] error: --input path does not exist: {resolved}")


def _apply_pii_ingest_policy(
    docs: list[Document],
    settings: Settings,
    detector: PIIDetector,
    audit: PIIAuditLog
) -> tuple[list[Document], list[Chunk]]:
    from ingest.pii import redact

    if settings.pii_mode == "redact":
        # Redact raw docs prior to chunking
        clean_docs = []
        for doc in docs:
            spans = detector.detect(doc.text)
            # Record audit values
            audit.record(
                tenant_id=doc.tenant_id,
                doc_id=doc.doc_id,
                chunk_id="doc_redaction",
                text=doc.text,
                spans=spans
            )
            clean_text = redact(doc.text, spans)
            clean_docs.append(doc.model_copy(update={"text": clean_text}))
        return clean_docs, []

    else:  # keep mode
        return docs, []


def _process_pdf(
    pdf_path: Path,
    *,
    settings: Settings,
    parser_registry,
    ingestor,
    manifest_store,
    detector: PIIDetector,
    audit: PIIAuditLog,
    prefixer,
    tenant_id: str,
    acl_tags: tuple[str, ...],
    collection_id: str,
) -> dict:
    """Run one PDF through parse → PII → chunk → contextual? → incremental ingest.

    Returns per-document stats used for progress lines and the final summary.
    Raises on failure; the caller decides whether to continue with other files.
    """
    doc_id = _doc_id_for(pdf_path)

    # Snapshot the previous manifest BEFORE ingesting, so we can report how
    # much work IncrementalIngestor skipped. Uses the same hashing functions
    # IncrementalIngestor applies, so the counts match its diff exactly.
    old = manifest_store.load(tenant_id, doc_id)
    prev_records = old.chunks if old else {}

    raw = pdf_path.read_bytes()
    parser_registry.guard_size(raw)
    parser = parser_registry.resolve(PDF_CONTENT_TYPE)
    docs = parser.parse(raw, pdf_path.name, PDF_CONTENT_TYPE,
                        doc_id=doc_id, tenant_id=tenant_id, acl_tags=acl_tags)
    if collection_id:
        docs = [d.model_copy(update={"collection_id": collection_id}) for d in docs]

    # Fail closed on PII policy errors.
    clean_docs, _ = _apply_pii_ingest_policy(docs, settings, detector, audit)

    from ingest.chunking import chunk_document

    chunks: list[Chunk] = []
    for doc in clean_docs:
        chunks.extend(chunk_document(doc))

    if settings.pii_mode == "keep":
        for ch in chunks:
            spans = detector.detect(ch.text)
            if spans:
                ch.metadata["pii_types"] = sorted({s.type for s in spans})
                audit.record(tenant_id=ch.tenant_id, doc_id=ch.doc_id,
                             chunk_id=ch.chunk_id, text=ch.text, spans=spans)

    if prefixer is not None:
        doc_by_id = {d.doc_id: d.text for d in clean_docs}
        chunks = prefixer.annotate(chunks, doc_by_id)

    from ingest.incremental import _hash, _meta_hash

    embedded = 0
    meta_only = 0
    for c in chunks:
        rec = prev_records.get(c.chunk_id)
        if rec is None or rec.embed_hash != _hash(c.embed_text):
            embedded += 1
        elif rec.meta_hash != _meta_hash(c):
            meta_only += 1
    unchanged = len(chunks) - embedded - meta_only

    acl = ACLContext(tenant_id=tenant_id, acl_tags=acl_tags)
    n_chunks = ingestor.ingest_document(tenant_id, doc_id, chunks, acl)
    return {"doc_id": doc_id, "chunks": n_chunks, "embedded": embedded,
            "meta_updated": meta_only, "unchanged": unchanged}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest PDF file(s) into the RAG stack "
                    "(parse → chunk → optional contextual prefixes → "
                    "incremental embed/upsert).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--tenant",
        default="public",
        help="Tenant that owns the ingested documents (default: public).",
    )
    parser.add_argument(
        "--acl-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="ACL tag to attach to every ingested chunk (repeatable).",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Optional collection id to group the documents under.",
    )
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="Add LLM-generated contextual prefixes to chunks before embedding.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, descend into subdirectories.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    from core.config import get_settings
    settings = get_settings()

    paths = _collect_pdf_paths(args.input, args.recursive)
    tenant_id = args.tenant
    acl_tags = tuple(args.acl_tag)
    collection_id = args.collection

    print(f"[ingest] {len(paths)} PDF(s) queued "
          f"(tenant={tenant_id!r}, collection={collection_id!r}, "
          f"contextual={args.contextual})")

    # --- Build components via core.registry factories ---
    from core.registry import (build_incremental_ingestor, build_manifest_store,
                               build_parser_registry, build_pii_detector)

    parser_registry = build_parser_registry(settings)
    ingestor = build_incremental_ingestor(settings)
    manifest_store = build_manifest_store(settings)
    detector = build_pii_detector(settings)
    audit = PIIAuditLog(settings)

    prefixer = None
    if args.contextual:
        from core.registry import build_generator
        from ingest.contextual import ContextualPrefixer

        generator = build_generator(role="context", settings=settings)
        prefixer = ContextualPrefixer(generator,
                                      cache_dir=settings.contextual_cache_dir,
                                      settings=settings)

    total_chunks = 0
    total_embedded = 0
    total_meta = 0
    total_unchanged = 0
    failures: list[tuple[Path, Exception]] = []

    for i, pdf_path in enumerate(paths, start=1):
        try:
            stats = _process_pdf(
                pdf_path,
                settings=settings,
                parser_registry=parser_registry,
                ingestor=ingestor,
                manifest_store=manifest_store,
                detector=detector,
                audit=audit,
                prefixer=prefixer,
                tenant_id=tenant_id,
                acl_tags=acl_tags,
                collection_id=collection_id,
            )
        except Exception as e:  # one bad PDF must not sink the batch
            print(f"[{i}/{len(paths)}] FAILED {pdf_path.name}: "
                  f"{type(e).__name__}: {e}")
            failures.append((pdf_path, e))
            continue
        total_chunks += stats["chunks"]
        total_embedded += stats["embedded"]
        total_meta += stats["meta_updated"]
        total_unchanged += stats["unchanged"]
        print(f"[{i}/{len(paths)}] ingested {pdf_path.name} → "
              f"{stats['chunks']} chunks "
              f"({stats['embedded']} embedded, {stats['meta_updated']} meta-updated, "
              f"{stats['unchanged']} unchanged)")

    print(f"[ingest] summary: {len(paths) - len(failures)}/{len(paths)} documents "
          f"ingested, {total_chunks} chunks indexed "
          f"({total_embedded} newly embedded/upserted, {total_meta} metadata-only "
          f"updates, {total_unchanged} skipped unchanged), {len(failures)} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
