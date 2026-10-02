from __future__ import annotations

import hashlib
import logging

from core.types import ACLContext, Chunk, ChunkRecord, DocManifest

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "v1"
_UPSERT_BATCH = 256

_TRACER = None


def _get_tracer():
    """Cached no-op-safe tracer so ingest spans cost nothing when disabled."""
    global _TRACER
    if _TRACER is None:
        from core.config import get_settings
        from observability.langfuse_tracing import Tracer

        _TRACER = Tracer(get_settings())
    return _TRACER


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _meta_hash(chunk: Chunk) -> str:
    key = (f"{chunk.title}:{chunk.tenant_id}:{sorted(chunk.acl_tags)}:"
           f"{chunk.collection_id}")
    return _hash(key)


def _meta_payload(chunk: Chunk) -> dict:
    """Every payload field derived from chunk metadata (must mirror
    qdrant_store._payload_from_chunk), so a meta-only change is fully applied."""
    return {
        "title": chunk.title,
        "collection_id": chunk.collection_id,
        "acl_tags": list(chunk.acl_tags),
        "acl_open": not bool(chunk.acl_tags),
    }


class IncrementalIngestor:
    """Diff a document's chunks against its manifest and apply the minimum work:
    embed+upsert new/changed chunks, metadata-update chunks whose only the meta
    changed, delete orphaned chunks. Manifest is saved LAST (after store writes),
    so a crash re-runs the same delta on retry (idempotent)."""

    def __init__(self, embedder, vector_store, sparse, manifest_store) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._sparse = sparse
        self._manifest = manifest_store

    def ensure_collection(self) -> None:
        """Create the vector store's collection if it doesn't exist yet, sized
        to the embedder's dimension — mirrors the CLI path (ingest/run.py).
        Idempotent but not free (get_collections + create_collection, plus two
        create_payload_index calls): call once per process, not per document."""
        self._store.ensure_collection(self._embedder.dimension)

    def ingest_document(self, tenant_id: str, doc_id: str,
                        chunks: list[Chunk], acl: ACLContext) -> int:
        tracer = _get_tracer()
        with tracer.span(
            "ingest.document", as_type="span",
            doc_id=doc_id, tenant_id=tenant_id, n_chunks=len(chunks),
        ) as s_doc:
            old = self._manifest.load(tenant_id, doc_id)
            old_chunks = old.chunks if old else {}

            new_records: dict[str, ChunkRecord] = {}
            to_embed: list[Chunk] = []
            to_meta: dict[str, dict] = {}
            meta_chunks: list[Chunk] = []

            for c in chunks:
                e_hash = _hash(c.embed_text)
                m_hash = _meta_hash(c)
                new_records[c.chunk_id] = ChunkRecord(
                    chunk_id=c.chunk_id, ordinal=c.ordinal,
                    embed_hash=e_hash, meta_hash=m_hash,
                )
                prev = old_chunks.get(c.chunk_id)
                if prev is None or prev.embed_hash != e_hash:
                    to_embed.append(c)
                elif prev.meta_hash != m_hash:
                    to_meta[c.chunk_id] = _meta_payload(c)
                    meta_chunks.append(c)

            to_delete = [cid for cid in old_chunks if cid not in new_records]

            try:
                if to_embed:
                    from core.config import get_settings

                    # First-run bootstrap: create the collection if absent
                    # (idempotent). Dimension must match the embedder.
                    ensure = getattr(self._store, "ensure_collection", None)
                    if ensure is not None:
                        ensure(get_settings().embed_dimension)

                    with tracer.span(
                        "ingest.embed_documents", as_type="embedding",
                        model=get_settings().embed_model, n_chunks=len(to_embed),
                    ) as s_emb:
                        vectors = self._embedder.embed_documents(
                            [c.embed_text for c in to_embed])
                        s_emb.update(n_chars=sum(len(c.embed_text) for c in to_embed))
                    embedded = [c.model_copy(update={"embedding": v})
                                for c, v in zip(to_embed, vectors)]
                    for i in range(0, len(embedded), _UPSERT_BATCH):
                        self._store.upsert(embedded[i : i + _UPSERT_BATCH])
                    self._sparse.add(embedded)
                if to_meta:
                    self._store.update_metadata(to_meta, acl)
                    # add() replaces in place by chunk_id, so the sparse copy
                    # picks up the new ACL/collection too.
                    self._sparse.add(meta_chunks)
                if to_delete:
                    self._store.delete(to_delete, acl)
                    self._sparse.delete(to_delete, acl)

                # D-ORDER: manifest only after store writes succeed.
                self._manifest.save(DocManifest(
                    tenant_id=tenant_id, doc_id=doc_id,
                    prompt_version=_PROMPT_VERSION, chunks=new_records,
                ))
            except Exception:
                if old is None:
                    # New doc: no manifest means no retry will reconcile, so
                    # best-effort remove whatever already landed.
                    try:
                        self._purge_doc(tenant_id, doc_id, acl, ())
                    except Exception:
                        logger.exception("cleanup after failed ingest of %s", doc_id)
                raise

            s_doc.update(
                embedded=len(to_embed),
                meta_updated=len(to_meta),
                deleted=len(to_delete),
                unchanged=len(new_records) - len(to_embed) - len(to_meta),
            )
        return len(new_records)

    def _purge_doc(self, tenant_id: str, doc_id: str, acl: ACLContext,
                   known_chunk_ids) -> int:
        """Remove a document from dense + sparse WITHOUT trusting the manifest:
        dense by (tenant_id, doc_id) payload filter (returns the chunk_ids it
        found), sparse by the union of those ids and any manifest-known ids.
        Best-effort per store so one failure doesn't strand the other."""
        ids = set(known_chunk_ids)
        errors: list[Exception] = []
        try:
            ids |= set(self._store.delete_by_doc(tenant_id, doc_id))
        except Exception as e:  # keep going: still purge sparse
            errors.append(e)
        if ids:
            try:
                self._sparse.delete(list(ids), acl)
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]
        return len(ids)

    def delete_document(self, tenant_id: str, doc_id: str, acl: ACLContext) -> int:
        old = self._manifest.load(tenant_id, doc_id)
        known = list(old.chunks.keys()) if old else []
        n = self._purge_doc(tenant_id, doc_id, acl, known)
        self._manifest.delete(tenant_id, doc_id)
        with _get_tracer().span(
            "ingest.delete_document", as_type="span",
            doc_id=doc_id, tenant_id=tenant_id, n_chunks=n,
        ):
            pass
        return n
