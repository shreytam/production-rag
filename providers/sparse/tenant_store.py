from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from core.types import ACLContext, Chunk, ScoredChunk
from providers.sparse.bm25 import BM25Retriever


class TenantSparseStore:
    """Persistent, per-tenant BM25. Each tenant's chunk list is pickled to its
    own file; the BM25 index is rebuilt on load. Mutations save immediately so a
    separate worker process and the API process share one on-disk source.

    SECURITY: `tenant_id` is caller-controlled (derived from JWT claims and only
    non-empty-validated — see core.types.ACLContext). The on-disk filename is
    derived from a SHA-256 hash of tenant_id rather than the raw value, so a
    tenant_id crafted to contain path-traversal sequences (e.g. "../../evil")
    cannot escape `index_dir`.
    """

    def __init__(self, index_dir: str = ".cache/sparse_tenants") -> None:
        self._dir = Path(index_dir)
        self._cache: dict[str, BM25Retriever] = {}

    def _path(self, tenant_id: str) -> Path:
        safe = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return self._dir / f"{safe}.pkl"

    def _retriever(self, tenant_id: str) -> BM25Retriever:
        if tenant_id in self._cache:
            return self._cache[tenant_id]
        r = BM25Retriever()
        path = self._path(tenant_id)
        if path.exists():
            chunks: list[Chunk] = pickle.loads(path.read_bytes())
            r.load_snapshot(tenant_id, chunks)
        self._cache[tenant_id] = r
        return r

    def _save(self, tenant_id: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        chunks = self._cache[tenant_id].snapshot(tenant_id)
        tmp = self._path(tenant_id).with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(chunks))
        tmp.replace(self._path(tenant_id))  # atomic

    def add(self, chunks: list[Chunk]) -> None:
        by_tenant: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_tenant.setdefault(c.tenant_id, []).append(c)
        for tenant_id, tenant_chunks in by_tenant.items():
            self._retriever(tenant_id).add(tenant_chunks)
            self._save(tenant_id)

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        self._retriever(acl.tenant_id).delete(chunk_ids, acl)
        self._save(acl.tenant_id)

    def search(self, query: str, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]:
        return self._retriever(acl.tenant_id).search(query, top_k, acl, collection_id=collection_id)

    def index(self, chunks: list[Chunk]) -> None:
        # Full (re)index: route through add so persistence + partitioning apply.
        self.add(chunks)
