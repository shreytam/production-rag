from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from core.types import ACLContext, Chunk, ScoredChunk
from providers.sparse.bm25 import BM25Retriever


class TenantSparseStore:
    """Persistent, per-tenant BM25. Each tenant's chunk list is stored as JSON in its
    own file; the BM25 index is rebuilt on load. Mutations save immediately so a
    separate worker process and the API process share one on-disk source.

    Cross-process consistency: readers stat() the snapshot on every search and
    reload when it changed; writers hold an exclusive per-tenant flock for the
    whole read-modify-write so concurrent writers never lose updates. JSON (not
    pickle) is used because the directory is shared between processes.

    SECURITY: `tenant_id` is caller-controlled (derived from JWT claims and only
    non-empty-validated — see core.types.ACLContext). The on-disk filename is
    derived from a SHA-256 hash of tenant_id rather than the raw value, so a
    tenant_id crafted to contain path-traversal sequences (e.g. "../../evil")
    cannot escape `index_dir`.
    """

    def __init__(self, index_dir: str = ".cache/sparse_tenants") -> None:
        self._dir = Path(index_dir)
        # tenant_id -> (file identity at load time, retriever)
        self._cache: dict[str, tuple[tuple[int, int, int] | None, BM25Retriever]] = {}

    def _path(self, tenant_id: str) -> Path:
        safe = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return self._dir / f"{safe}.json"

    def _stat_id(self, tenant_id: str) -> tuple[int, int, int] | None:
        try:
            st = self._path(tenant_id).stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _retriever(self, tenant_id: str) -> BM25Retriever:
        """Return the tenant's retriever, reloading if the snapshot changed on disk."""
        ident = self._stat_id(tenant_id)
        cached = self._cache.get(tenant_id)
        if cached is not None and cached[0] == ident:
            return cached[1]
        r = BM25Retriever()
        if ident is not None:
            try:
                raw = json.loads(self._path(tenant_id).read_text(encoding="utf-8"))
            except FileNotFoundError:  # deleted between stat and read
                raw, ident = [], None
            r.load_snapshot(tenant_id, [Chunk.model_validate(c) for c in raw])
        self._cache[tenant_id] = (ident, r)
        return r

    @contextmanager
    def _locked(self, tenant_id: str) -> Iterator[None]:
        """Exclusive per-tenant cross-process lock around a read-modify-write."""
        self._dir.mkdir(parents=True, exist_ok=True)
        lock = self._path(tenant_id).with_suffix(".lock")
        with open(lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _save(self, tenant_id: str, retriever: BM25Retriever) -> None:
        chunks = retriever.snapshot(tenant_id)
        path = self._path(tenant_id)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks]), encoding="utf-8"
        )
        tmp.replace(path)  # atomic
        self._cache[tenant_id] = (self._stat_id(tenant_id), retriever)

    def add(self, chunks: list[Chunk]) -> None:
        by_tenant: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_tenant.setdefault(c.tenant_id, []).append(c)
        for tenant_id, tenant_chunks in by_tenant.items():
            with self._locked(tenant_id):
                r = self._retriever(tenant_id)  # fresh from disk under the lock
                r.add(tenant_chunks)
                self._save(tenant_id, r)

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        with self._locked(acl.tenant_id):
            r = self._retriever(acl.tenant_id)
            r.delete(chunk_ids, acl)
            self._save(acl.tenant_id, r)

    def search(self, query: str, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]:
        return self._retriever(acl.tenant_id).search(query, top_k, acl, collection_id=collection_id)

    def index(self, chunks: list[Chunk]) -> None:
        # Full (re)index: route through add so persistence + partitioning apply.
        self.add(chunks)
