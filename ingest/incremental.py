from __future__ import annotations

import hashlib

from core.types import ACLContext, Chunk, ChunkRecord, DocManifest

_PROMPT_VERSION = "v1"


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _meta_hash(chunk: Chunk) -> str:
    key = f"{chunk.title}:{chunk.tenant_id}:{sorted(chunk.acl_tags)}"
    return _hash(key)


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

    def ingest_document(self, tenant_id: str, doc_id: str,
                        chunks: list[Chunk], acl: ACLContext) -> int:
        old = self._manifest.load(tenant_id, doc_id)
        old_chunks = old.chunks if old else {}

        new_records: dict[str, ChunkRecord] = {}
        to_embed: list[Chunk] = []
        to_meta: dict[str, dict] = {}

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
                to_meta[c.chunk_id] = {"title": c.title}

        to_delete = [cid for cid in old_chunks if cid not in new_records]

        if to_embed:
            vectors = self._embedder.embed_documents([c.embed_text for c in to_embed])
            embedded = [c.model_copy(update={"embedding": v})
                        for c, v in zip(to_embed, vectors)]
            self._store.upsert(embedded)
            self._sparse.add(embedded)
        if to_meta:
            self._store.update_metadata(to_meta, acl)
        if to_delete:
            self._store.delete(to_delete, acl)
            self._sparse.delete(to_delete, acl)

        # D-ORDER: manifest only after store writes succeed.
        self._manifest.save(DocManifest(
            tenant_id=tenant_id, doc_id=doc_id,
            prompt_version=_PROMPT_VERSION, chunks=new_records,
        ))
        return len(new_records)
