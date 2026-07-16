# Product Ingestion — API-driven, Async, Incremental — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an API-driven, asynchronous, multi-format document ingestion path (upload → parse → PII → chunk → embed → store, scoped to the caller's tenant) so the existing `/query` retrieves over user-uploaded documents.

**Architecture:** New FastAPI routes accept an upload, persist the raw bytes via a `BlobStore`, record a row in a Postgres-backed document registry (`processing|ready|failed`), and enqueue an `arq` job on the existing Redis. A worker parses the file behind a `DocumentParser` interface, then drives the SP8 `IncrementalIngestor` (content-hash delta over dense store + manifest) extended to also update a per-tenant, persistent BM25 index. All model calls route through one OpenAI-compatible base_url/api_key (reranker excluded).

**Tech Stack:** Python 3.11–3.13, FastAPI, `arq` (Redis queue), `redis`, `psycopg` (registry), `qdrant-client`, `rank-bm25`, `unstructured` (parser), Pydantic v2, pytest, `uv`.

## Global Constraints

- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- `core/registry.py` is the ONLY module that names concrete implementation classes; everything else depends on the `core/interfaces.py` Protocols.
- Tenant isolation is fail-closed: chunk `tenant_id`/`acl_tags` derive from the verified `Principal` captured at upload, NEVER from file content. Cross-tenant delete/update/read must be structurally impossible, not a post-filter.
- New heavy/optional imports (`unstructured`, `arq`, `psycopg`) stay LOCAL to the function that needs them so importing a module is cheap and offline-safe.
- All tests in this plan run OFFLINE with fakes — no network, no live Qdrant/Redis/Postgres, no live models.
- Dependencies are added to the `app` extra in `pyproject.toml`.

---

### Task 1: Single OpenAI-compatible model router in config

**Files:**
- Modify: `core/config.py`
- Test: `tests/test_ingest_router_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings.llm_base_url: str`, `Settings.llm_api_key: str`. After model validation, any of `embed_base_url`, `gen_base_url`, `context_base_url`, `judge_base_url` left at the default NIM URL and any of `embed_api_key`/`gen_api_key`/`context_api_key`/`judge_api_key` left empty fall back to the router values. The reranker fields are untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_router_config.py
from core.config import Settings


def test_router_fills_all_model_roles_but_not_reranker():
    s = Settings(
        llm_base_url="https://router.example/v1",
        llm_api_key="router-key",
        nvidia_api_key="",  # ensure no other fallback masks the router
    )
    for url in (s.embed_base_url, s.gen_base_url, s.context_base_url, s.judge_base_url):
        assert url == "https://router.example/v1"
    for key in (s.embed_api_key, s.gen_api_key, s.context_api_key, s.judge_api_key):
        assert key == "router-key"
    # Reranker is NOT routed through the OpenAI-compatible router.
    assert s.reranker_nim_base_url != "https://router.example/v1"


def test_explicit_role_override_beats_router():
    s = Settings(
        llm_base_url="https://router.example/v1",
        llm_api_key="router-key",
        gen_base_url="https://custom.example/v1",
        gen_api_key="custom-key",
    )
    assert s.gen_base_url == "https://custom.example/v1"
    assert s.gen_api_key == "custom-key"
    assert s.embed_base_url == "https://router.example/v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest_router_config.py -v`
Expected: FAIL (`TypeError`/validation error — `llm_base_url` is not a field).

- [ ] **Step 3: Add fields and a fallback validator**

In `core/config.py`, add the fields near the "Shared keys" block:

```python
    # --- OpenAI-compatible model router (one base_url + key for all roles) ---
    # Any model role left at the NIM default url / empty key inherits these.
    # The reranker is intentionally excluded (no OpenAI-standard rerank endpoint).
    llm_base_url: str = ""
    llm_api_key: str = ""
```

Add a validator that runs BEFORE `_fill_key_fallbacks` (define it above that method so declaration order makes it run first):

```python
    @model_validator(mode="after")
    def _apply_llm_router(self) -> "Settings":
        """Point every model role at one OpenAI-compatible router unless the role
        was explicitly overridden. A role url still at the NIM default, or a role
        with an empty base url, adopts llm_base_url; empty role keys adopt
        llm_api_key. The reranker is deliberately not routed."""
        if self.llm_base_url:
            for field in ("embed_base_url", "gen_base_url", "context_base_url", "judge_base_url"):
                if getattr(self, field) in ("", NIM_BASE_URL):
                    setattr(self, field, self.llm_base_url)
        if self.llm_api_key:
            for field in ("embed_api_key", "gen_api_key", "context_api_key", "judge_api_key"):
                if not getattr(self, field):
                    setattr(self, field, self.llm_api_key)
        return self
```

Note: `_apply_llm_router` must be declared ABOVE `_fill_key_fallbacks` so the router fills base urls first; `_fill_key_fallbacks` then still fills any key the router did not.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest_router_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_ingest_router_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(config): route all model roles through one OpenAI-compatible base_url/key" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: VectorStore delete + update_metadata (tenant-scoped)

**Files:**
- Modify: `core/interfaces.py`
- Modify: `providers/vectorstores/qdrant_store.py`
- Modify: `providers/vectorstores/pgvector_store.py`
- Modify: `tests/_fakes.py`
- Test: `tests/test_store_mutations.py`

**Interfaces:**
- Consumes: `ACLContext`, `Chunk` (from `core.types`).
- Produces: on the `VectorStore` Protocol and every implementation —
  `delete(self, chunk_ids: list[str], acl: ACLContext) -> None` and
  `update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None`.
  Both are tenant-scoped: a chunk whose `tenant_id` differs from `acl.tenant_id`
  is never deleted or mutated. `InMemoryVectorStore` (test fake) implements both.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_mutations.py
from core.types import ACLContext, Chunk
from tests._fakes import InMemoryVectorStore, vec


def _chunk(cid, tenant, text="hello world"):
    return Chunk(chunk_id=cid, doc_id="d1", text=text, tenant_id=tenant,
                 embedding=vec(text))


def test_delete_is_tenant_scoped():
    store = InMemoryVectorStore()
    store.upsert([_chunk("c1", "t1"), _chunk("c2", "t2")])
    store.delete(["c1", "c2"], ACLContext(tenant_id="t1"))
    remaining = {c.chunk_id for c in store.chunks}
    assert remaining == {"c2"}  # c2 belongs to t2, untouched by a t1 caller


def test_update_metadata_is_tenant_scoped():
    store = InMemoryVectorStore()
    store.upsert([_chunk("c1", "t1"), _chunk("c2", "t2")])
    store.update_metadata({"c1": {"title": "X"}, "c2": {"title": "Y"}},
                          ACLContext(tenant_id="t1"))
    by_id = {c.chunk_id: c for c in store.chunks}
    assert by_id["c1"].title == "X"
    assert by_id["c2"].title != "Y"  # cross-tenant update refused
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store_mutations.py -v`
Expected: FAIL (`AttributeError: 'InMemoryVectorStore' object has no attribute 'delete'`).

- [ ] **Step 3: Extend the protocol, the fake, and both real stores**

In `core/interfaces.py`, add to the `VectorStore` Protocol:

```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None: ...

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None: ...
```

In `tests/_fakes.py`, add to `InMemoryVectorStore`:

```python
    def delete(self, chunk_ids, acl):
        wanted = set(chunk_ids)
        self.chunks = [
            c for c in self.chunks
            if not (c.chunk_id in wanted and c.tenant_id == acl.tenant_id)
        ]

    def update_metadata(self, updates, acl):
        for c in self.chunks:
            if c.tenant_id != acl.tenant_id:
                continue
            payload = updates.get(c.chunk_id)
            if payload and "title" in payload:
                c.title = payload["title"]
```

In `providers/vectorstores/qdrant_store.py`, add methods on `QdrantVectorStore`:

```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        ids = [_chunk_uuid(cid) for cid in chunk_ids]
        combined = qm.Filter(must=[qm.HasIdCondition(has_id=ids), qdrant_filter(acl)])
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=combined),
        )

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        for chunk_id, payload in updates.items():
            pt = _chunk_uuid(chunk_id)
            combined = qm.Filter(must=[qm.HasIdCondition(has_id=[pt]), qdrant_filter(acl)])
            self._client.set_payload(
                collection_name=self._collection,
                payload=payload,
                points=qm.FilterSelector(filter=combined),
            )
```

In `providers/vectorstores/pgvector_store.py`, add methods on `PgVectorStore` (use the existing ACL where-clause helper this file already imports for `search`; if it imports `pg_where` from `retrieval.acl`, reuse it):

```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        from retrieval.acl import pg_where
        where_clause, params = pg_where(acl)
        sql = f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s) AND {where_clause}"
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [list(chunk_ids), *params])

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        from retrieval.acl import pg_where
        where_clause, params = pg_where(acl)
        with self._pool.connection() as conn, conn.cursor() as cur:
            for chunk_id, payload in updates.items():
                if "title" not in payload:
                    continue
                sql = (f"UPDATE {self._table} SET title = %s "
                       f"WHERE chunk_id = %s AND {where_clause}")
                cur.execute(sql, [payload["title"], chunk_id, *params])
```

Note: if `pgvector_store.py` uses different attribute names (`self._table`, `self._pool`) or a different ACL helper, match this file's existing `search()` implementation exactly rather than the names above.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store_mutations.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/interfaces.py providers/vectorstores/qdrant_store.py providers/vectorstores/pgvector_store.py tests/_fakes.py tests/test_store_mutations.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add tenant-scoped delete and update_metadata to vector stores" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Document manifest + JSONL manifest store

**Files:**
- Modify: `core/types.py`
- Modify: `core/interfaces.py`
- Modify: `core/config.py`
- Modify: `core/registry.py`
- Create: `providers/manifest/__init__.py`
- Create: `providers/manifest/jsonl_store.py`
- Test: `tests/test_manifest_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ChunkRecord(chunk_id, ordinal, embed_hash, meta_hash)`,
  `DocManifest(tenant_id, doc_id, prompt_version, chunks: dict[str, ChunkRecord])`,
  `ManifestStore` Protocol with `load(tenant_id, doc_id) -> DocManifest | None`,
  `save(manifest) -> None`, `delete(tenant_id, doc_id) -> None`,
  `JsonlManifestStore(manifest_dir)`, and `build_manifest_store(settings) -> ManifestStore`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_store.py
from core.types import ChunkRecord, DocManifest
from providers.manifest.jsonl_store import JsonlManifestStore


def test_save_and_load_roundtrip(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    m = DocManifest(tenant_id="t1", doc_id="d1", prompt_version="v1",
                    chunks={"c1": ChunkRecord(chunk_id="c1", ordinal=0,
                                              embed_hash="h1", meta_hash="h2")})
    store.save(m)
    loaded = store.load("t1", "d1")
    assert loaded is not None
    assert loaded.chunks["c1"].embed_hash == "h1"


def test_load_missing_returns_none(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    assert store.load("t1", "nope") is None


def test_tenant_mismatch_fails_closed(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    m = DocManifest(tenant_id="t1", doc_id="d1", prompt_version="v1", chunks={})
    store.save(m)
    assert store.load("t2", "d1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest_store.py -v`
Expected: FAIL (`ModuleNotFoundError: providers.manifest`).

- [ ] **Step 3: Add types, protocol, config, store, builder**

In `core/types.py` add:

```python
class ChunkRecord(BaseModel):
    chunk_id: str
    ordinal: int
    embed_hash: str
    meta_hash: str


class DocManifest(BaseModel):
    tenant_id: str
    doc_id: str
    prompt_version: str = "v1"
    chunks: dict[str, ChunkRecord] = Field(default_factory=dict)
```

In `core/interfaces.py` add (import `DocManifest` in the `from core.types import (...)` block):

```python
@runtime_checkable
class ManifestStore(Protocol):
    def load(self, tenant_id: str, doc_id: str) -> "DocManifest | None": ...
    def save(self, manifest: "DocManifest") -> None: ...
    def delete(self, tenant_id: str, doc_id: str) -> None: ...
```

In `core/config.py` add a field:

```python
    manifest_dir: str = ".cache/manifest"
```

Create `providers/manifest/__init__.py` (empty). Create `providers/manifest/jsonl_store.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from core.types import DocManifest


class JsonlManifestStore:
    def __init__(self, manifest_dir: str = ".cache/manifest") -> None:
        self.manifest_dir = Path(manifest_dir)

    def _path_for(self, tenant_id: str, doc_id: str) -> Path:
        safe = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        return self.manifest_dir / tenant_id / f"{safe}.json"

    def load(self, tenant_id: str, doc_id: str) -> DocManifest | None:
        path = self._path_for(tenant_id, doc_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("tenant_id") != tenant_id or data.get("doc_id") != doc_id:
                return None
            return DocManifest.model_validate(data)
        except Exception:
            return None

    def save(self, manifest: DocManifest) -> None:
        path = self._path_for(manifest.tenant_id, manifest.doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(manifest.model_dump(), indent=2))
            os.replace(tmp, path)  # atomic
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    def delete(self, tenant_id: str, doc_id: str) -> None:
        path = self._path_for(tenant_id, doc_id)
        if path.exists():
            path.unlink()
```

In `core/registry.py` add (and import `ManifestStore` in the `core.interfaces` import line):

```python
def build_manifest_store(settings: Settings | None = None) -> ManifestStore:
    s = settings or get_settings()
    from providers.manifest.jsonl_store import JsonlManifestStore

    return JsonlManifestStore(manifest_dir=s.manifest_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_manifest_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/types.py core/interfaces.py core/config.py core/registry.py providers/manifest tests/test_manifest_store.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add DocManifest schema and JsonlManifestStore" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Incremental add/delete on the BM25 sparse retriever

**Files:**
- Modify: `core/interfaces.py`
- Modify: `providers/sparse/bm25.py`
- Test: `tests/test_bm25_incremental.py`

**Interfaces:**
- Consumes: `Chunk`, `ACLContext`.
- Produces: on the `SparseRetriever` Protocol and `BM25Retriever` —
  `add(self, chunks: list[Chunk]) -> None` (merge into the per-tenant index,
  rebuilding that tenant's `BM25Okapi`) and
  `delete(self, chunk_ids: list[str], acl: ACLContext) -> None` (drop from the
  tenant's index, rebuild). `BM25Retriever.snapshot(tenant_id)` /
  `BM25Retriever.load_snapshot(tenant_id, chunks)` expose the per-tenant chunk
  list for persistence (Task 5). Existing `index()` is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25_incremental.py
from core.types import ACLContext, Chunk
from providers.sparse.bm25 import BM25Retriever


def _c(cid, tenant, text):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, tenant_id=tenant)


def test_add_makes_chunk_searchable():
    r = BM25Retriever()
    r.add([_c("d1::0", "t1", "alpha beta")])
    r.add([_c("d2::0", "t1", "gamma delta")])
    hits = r.search("gamma", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits][:1] == ["d2::0"]


def test_add_is_tenant_partitioned():
    r = BM25Retriever()
    r.add([_c("d1::0", "t1", "alpha")])
    r.add([_c("d2::0", "t2", "alpha")])
    hits = r.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert {h.chunk.tenant_id for h in hits} == {"t1"}


def test_delete_removes_chunk():
    r = BM25Retriever()
    r.add([_c("d1::0", "t1", "alpha beta"), _c("d1::1", "t1", "alpha gamma")])
    r.delete(["d1::0"], ACLContext(tenant_id="t1"))
    hits = r.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bm25_incremental.py -v`
Expected: FAIL (`AttributeError: 'BM25Retriever' object has no attribute 'add'`).

- [ ] **Step 3: Extend protocol and implementation**

In `core/interfaces.py`, add to the `SparseRetriever` Protocol:

```python
    def add(self, chunks: list[Chunk]) -> None: ...

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None: ...
```

In `providers/sparse/bm25.py`, add to `BM25Retriever`:

```python
    def _rebuild_tenant(self, tenant_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            self._indices.pop(tenant_id, None)
            return
        corpus = [_tokenize(c.embed_text) for c in chunks]
        self._indices[tenant_id] = (BM25Okapi(corpus), chunks)

    def add(self, chunks: list[Chunk]) -> None:
        by_tenant: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_tenant.setdefault(c.tenant_id, []).append(c)
        for tenant_id, new_chunks in by_tenant.items():
            existing = list(self._indices.get(tenant_id, (None, []))[1])
            existing_ids = {c.chunk_id for c in existing}
            merged = existing + [c for c in new_chunks if c.chunk_id not in existing_ids]
            # replace chunks whose id already existed (content may have changed)
            new_by_id = {c.chunk_id: c for c in new_chunks}
            merged = [new_by_id.get(c.chunk_id, c) for c in merged]
            self._rebuild_tenant(tenant_id, merged)

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        entry = self._indices.get(acl.tenant_id)
        if entry is None:
            return
        drop = set(chunk_ids)
        kept = [c for c in entry[1] if c.chunk_id not in drop]
        self._rebuild_tenant(acl.tenant_id, kept)

    def snapshot(self, tenant_id: str) -> list[Chunk]:
        entry = self._indices.get(tenant_id)
        return list(entry[1]) if entry else []

    def load_snapshot(self, tenant_id: str, chunks: list[Chunk]) -> None:
        self._rebuild_tenant(tenant_id, list(chunks))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bm25_incremental.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/interfaces.py providers/sparse/bm25.py tests/test_bm25_incremental.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add incremental add/delete and per-tenant snapshots to BM25" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: Persistent per-tenant sparse index + query-time resolution

**Files:**
- Modify: `core/config.py`
- Modify: `core/registry.py`
- Create: `providers/sparse/tenant_store.py`
- Modify: `core/pipeline.py`
- Test: `tests/test_tenant_sparse_store.py`

**Interfaces:**
- Consumes: `BM25Retriever` (Task 4), `ACLContext`, `Chunk`.
- Produces: `TenantSparseStore(index_dir)` implementing the `SparseRetriever`
  Protocol PLUS persistence: `search`/`add`/`delete` operate on the tenant
  resolved from `acl.tenant_id`, loading that tenant's pickle lazily and saving
  after each mutation. `build_tenant_sparse_store(settings) -> TenantSparseStore`.
  `core.pipeline.build(...)` uses `build_tenant_sparse_store` for `version="full"`
  instead of a corpus-bound pickle, so retrieval selects the tenant's index at
  query time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tenant_sparse_store.py
from core.types import ACLContext, Chunk
from providers.sparse.tenant_store import TenantSparseStore


def _c(cid, tenant, text):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, tenant_id=tenant)


def test_add_persists_across_instances(tmp_path):
    s1 = TenantSparseStore(index_dir=str(tmp_path))
    s1.add([_c("d1::0", "t1", "alpha beta")])
    # A fresh instance (new process) loads the persisted tenant index.
    s2 = TenantSparseStore(index_dir=str(tmp_path))
    hits = s2.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::0"]


def test_delete_persists(tmp_path):
    s = TenantSparseStore(index_dir=str(tmp_path))
    s.add([_c("d1::0", "t1", "alpha"), _c("d1::1", "t1", "beta")])
    s.delete(["d1::0"], ACLContext(tenant_id="t1"))
    s2 = TenantSparseStore(index_dir=str(tmp_path))
    hits = s2.search("alpha beta", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::1"]


def test_tenants_isolated_on_disk(tmp_path):
    s = TenantSparseStore(index_dir=str(tmp_path))
    s.add([_c("d1::0", "t1", "shared")])
    s.add([_c("d2::0", "t2", "shared")])
    hits = s.search("shared", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert {h.chunk.tenant_id for h in hits} == {"t1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tenant_sparse_store.py -v`
Expected: FAIL (`ModuleNotFoundError: providers.sparse.tenant_store`).

- [ ] **Step 3: Implement the store, config, builder, and pipeline wiring**

In `core/config.py` add:

```python
    tenant_sparse_dir: str = ".cache/sparse_tenants"
```

Create `providers/sparse/tenant_store.py`:

```python
from __future__ import annotations

import pickle
from pathlib import Path

from core.types import ACLContext, Chunk, ScoredChunk
from providers.sparse.bm25 import BM25Retriever


class TenantSparseStore:
    """Persistent, per-tenant BM25. Each tenant's chunk list is pickled to its
    own file; the BM25 index is rebuilt on load. Mutations save immediately so a
    separate worker process and the API process share one on-disk source."""

    def __init__(self, index_dir: str = ".cache/sparse_tenants") -> None:
        self._dir = Path(index_dir)
        self._cache: dict[str, BM25Retriever] = {}

    def _path(self, tenant_id: str) -> Path:
        return self._dir / f"{tenant_id}.pkl"

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

    def search(self, query: str, top_k: int, acl: ACLContext) -> list[ScoredChunk]:
        return self._retriever(acl.tenant_id).search(query, top_k, acl)

    def index(self, chunks: list[Chunk]) -> None:
        # Full (re)index: route through add so persistence + partitioning apply.
        self.add(chunks)
```

In `core/registry.py` add:

```python
def build_tenant_sparse_store(settings: Settings | None = None):
    s = settings or get_settings()
    from providers.sparse.tenant_store import TenantSparseStore

    return TenantSparseStore(index_dir=s.tenant_sparse_dir)
```

In `core/pipeline.py`, in `build()` under `elif version == "full":`, replace the
`sparse = build_sparse_retriever(s, resolved_corpus)` line and its emptiness
check with:

```python
        from core.registry import build_tenant_sparse_store

        sparse = build_tenant_sparse_store(s)
        reranker = build_reranker(s)
        retriever = HybridRetriever(embedder, store, sparse, reranker, rrf_k=s.rrf_k)
```

Remove the now-dead `is_empty` / `HybridIndexError` branch for `version="full"`
(the tenant store is always present and simply returns no hits for an unknown
tenant). Keep the `HybridIndexError` class definition — other callers/tests may
import it — but it is no longer raised here.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tenant_sparse_store.py -v`
Expected: PASS

Then run the pipeline tests to confirm the wiring change is intact:
Run: `uv run pytest tests/test_pipeline_integration.py -v`
Expected: PASS (adjust any test that constructed a corpus-bound sparse index to use `TenantSparseStore`).

- [ ] **Step 5: Commit**

```bash
git add core/config.py core/registry.py providers/sparse/tenant_store.py core/pipeline.py tests/test_tenant_sparse_store.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): persistent per-tenant sparse index resolved at query time" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: IncrementalIngestor (dense + sparse + manifest delta)

**Files:**
- Create: `ingest/incremental.py`
- Modify: `core/registry.py`
- Test: `tests/test_incremental_ingestor.py`

**Interfaces:**
- Consumes: `Embedder`, `VectorStore` (with `delete`/`update_metadata` — Task 2),
  `SparseRetriever` (with `add`/`delete` — Task 4/5), `ManifestStore` (Task 3),
  `Chunk`, `ACLContext`, `DocManifest`, `ChunkRecord`.
- Produces: `IncrementalIngestor(embedder, vector_store, sparse, manifest_store)`
  with `ingest_document(tenant_id: str, doc_id: str, chunks: list[Chunk], acl: ACLContext) -> int`
  returning the number of chunks now indexed for the document. Idempotent by
  `doc_id` + content hash. `build_incremental_ingestor(settings) -> IncrementalIngestor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incremental_ingestor.py
from core.types import ACLContext, Chunk
from ingest.incremental import IncrementalIngestor
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _c(cid, ordinal, text, tenant="t1"):
    return Chunk(chunk_id=cid, doc_id="d1", text=text, tenant_id=tenant, ordinal=ordinal)


def _ingestor(tmp_path):
    return IncrementalIngestor(
        FakeEmbedder(), InMemoryVectorStore(), BM25Retriever(),
        JsonlManifestStore(manifest_dir=str(tmp_path)),
    )


def test_first_ingest_embeds_and_indexes(tmp_path):
    ing = _ingestor(tmp_path)
    n = ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], ACLContext(tenant_id="t1"))
    assert n == 1
    assert len(ing._store.chunks) == 1
    assert ing._store.chunks[0].embedding is not None


def test_reingest_unchanged_is_noop(tmp_path):
    ing = _ingestor(tmp_path)
    acl = ACLContext(tenant_id="t1")
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], acl)
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "alpha")], acl)
    assert len(ing._store.chunks) == 1  # not duplicated


def test_removed_chunk_is_deleted(tmp_path):
    ing = _ingestor(tmp_path)
    acl = ACLContext(tenant_id="t1")
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "a"), _c("d1::1", 1, "b")], acl)
    ing.ingest_document("t1", "d1", [_c("d1::0", 0, "a")], acl)  # d1::1 dropped
    ids = {c.chunk_id for c in ing._store.chunks}
    assert ids == {"d1::0"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_incremental_ingestor.py -v`
Expected: FAIL (`ModuleNotFoundError: ingest.incremental`).

- [ ] **Step 3: Implement the ingestor and builder**

Create `ingest/incremental.py`:

```python
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
```

In `core/registry.py` add:

```python
def build_incremental_ingestor(settings: Settings | None = None):
    s = settings or get_settings()
    from ingest.incremental import IncrementalIngestor

    return IncrementalIngestor(
        build_embedder(s),
        build_vector_store(s),
        build_tenant_sparse_store(s),
        build_manifest_store(s),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_incremental_ingestor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/incremental.py core/registry.py tests/test_incremental_ingestor.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add IncrementalIngestor with content-hash delta over dense+sparse" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: DocumentParser subsystem (multi-format, allowlist, size guard)

**Files:**
- Modify: `core/config.py`
- Create: `ingest/parsers/__init__.py`
- Create: `ingest/parsers/base.py`
- Create: `ingest/parsers/plain_text.py`
- Create: `ingest/parsers/unstructured_parser.py`
- Modify: `core/registry.py`
- Test: `tests/test_document_parser.py`

**Interfaces:**
- Consumes: `Document` (from `core.types`).
- Produces: `ParserError(Exception)`; `DocumentParser` Protocol with
  `parse(self, raw: bytes, filename: str, content_type: str, *, doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]`;
  `PlainTextParser`; `UnstructuredParser` (lazy import);
  `ParserRegistry(allowed_types, max_bytes)` with
  `resolve(content_type) -> DocumentParser` (raises `ParserError` if type not in
  `allowed_types`) and `guard_size(raw)` (raises `ParserError` if over `max_bytes`).
  `build_parser_registry(settings) -> ParserRegistry`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_document_parser.py
import pytest

from ingest.parsers.base import ParserError, ParserRegistry
from ingest.parsers.plain_text import PlainTextParser


def test_plain_text_parser_makes_one_document():
    docs = PlainTextParser().parse(
        b"hello world", "note.txt", "text/plain",
        doc_id="d1", tenant_id="t1", acl_tags=(),
    )
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].doc_id == "d1"
    assert docs[0].tenant_id == "t1"


def test_registry_rejects_disallowed_type():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    with pytest.raises(ParserError):
        reg.resolve("application/x-evil")


def test_registry_rejects_oversize():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=4)
    with pytest.raises(ParserError):
        reg.guard_size(b"12345")


def test_registry_resolves_plain_text():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    assert isinstance(reg.resolve("text/plain"), PlainTextParser)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_document_parser.py -v`
Expected: FAIL (`ModuleNotFoundError: ingest.parsers`).

- [ ] **Step 3: Implement parsers, registry, config, builder**

In `core/config.py` add:

```python
    # --- Product ingest (API upload) ---
    ingest_allowed_types: str = (
        "text/plain,text/markdown,application/pdf,"
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
        "text/html"
    )
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MiB
```

Create `ingest/parsers/__init__.py` (empty). Create `ingest/parsers/base.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import Document


class ParserError(Exception):
    """Raised on unsupported type, oversize upload, or unparseable content."""


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]: ...


class ParserRegistry:
    def __init__(self, allowed_types: set[str], max_bytes: int) -> None:
        self._allowed = set(allowed_types)
        self._max_bytes = max_bytes

    def guard_size(self, raw: bytes) -> None:
        if len(raw) > self._max_bytes:
            raise ParserError(f"upload exceeds max_upload_bytes ({self._max_bytes})")

    def resolve(self, content_type: str) -> DocumentParser:
        if content_type not in self._allowed:
            raise ParserError(f"unsupported content_type: {content_type}")
        from ingest.parsers.plain_text import PlainTextParser
        if content_type in ("text/plain", "text/markdown"):
            return PlainTextParser()
        from ingest.parsers.unstructured_parser import UnstructuredParser
        return UnstructuredParser()
```

Create `ingest/parsers/plain_text.py`:

```python
from __future__ import annotations

from core.types import Document


class PlainTextParser:
    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]:
        text = raw.decode("utf-8", errors="replace")
        return [Document(doc_id=doc_id, text=text, tenant_id=tenant_id,
                         acl_tags=acl_tags, title=filename, source=f"upload:{content_type}")]
```

Create `ingest/parsers/unstructured_parser.py` (heavy import stays local; not exercised offline):

```python
from __future__ import annotations

from core.types import Document
from ingest.parsers.base import ParserError


class UnstructuredParser:
    """Rich multi-format parser (PDF/Office/HTML/OCR) via `unstructured`."""

    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]:
        try:
            from unstructured.partition.auto import partition  # local heavy import
        except ImportError as e:  # pragma: no cover
            raise ParserError("unstructured not installed") from e
        import io
        elements = partition(file=io.BytesIO(raw), metadata_filename=filename)
        text = "\n\n".join(str(el) for el in elements if str(el).strip())
        if not text:
            raise ParserError(f"no extractable text in {filename}")
        return [Document(doc_id=doc_id, text=text, tenant_id=tenant_id,
                         acl_tags=acl_tags, title=filename, source=f"upload:{content_type}")]
```

In `core/registry.py` add:

```python
def build_parser_registry(settings: Settings | None = None):
    s = settings or get_settings()
    from ingest.parsers.base import ParserRegistry

    allowed = {t.strip() for t in s.ingest_allowed_types.split(",") if t.strip()}
    return ParserRegistry(allowed_types=allowed, max_bytes=s.max_upload_bytes)
```

Add `unstructured>=0.15` to the `app` extra in `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_document_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/config.py ingest/parsers core/registry.py pyproject.toml tests/test_document_parser.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add DocumentParser subsystem with allowlist and size guard" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 8: BlobStore for raw uploads

**Files:**
- Modify: `core/config.py`
- Modify: `core/interfaces.py`
- Create: `providers/blobstore/__init__.py`
- Create: `providers/blobstore/local_disk.py`
- Modify: `core/registry.py`
- Test: `tests/test_blobstore.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BlobStore` Protocol with `put(key: str, data: bytes) -> None`,
  `get(key: str) -> bytes`, `delete(key: str) -> None`; `LocalDiskBlobStore(root)`;
  `build_blob_store(settings) -> BlobStore`. `get` on a missing key raises
  `KeyError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blobstore.py
import pytest

from providers.blobstore.local_disk import LocalDiskBlobStore


def test_put_get_roundtrip(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    store.put("tenant1/doc1.bin", b"payload")
    assert store.get("tenant1/doc1.bin") == b"payload"


def test_get_missing_raises(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    with pytest.raises(KeyError):
        store.get("nope")


def test_delete_removes(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    store.put("k", b"x")
    store.delete("k")
    with pytest.raises(KeyError):
        store.get("k")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_blobstore.py -v`
Expected: FAIL (`ModuleNotFoundError: providers.blobstore`).

- [ ] **Step 3: Implement protocol, store, config, builder**

In `core/config.py` add:

```python
    blob_store_root: str = ".cache/uploads"
```

In `core/interfaces.py` add:

```python
@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
```

Create `providers/blobstore/__init__.py` (empty). Create `providers/blobstore/local_disk.py`:

```python
from __future__ import annotations

from pathlib import Path


class LocalDiskBlobStore:
    def __init__(self, root: str = ".cache/uploads") -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self._root / key).resolve()
        if not str(p).startswith(str(self._root.resolve())):
            raise KeyError(f"illegal blob key: {key}")  # path traversal guard
        return p

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
```

In `core/registry.py` add (import `BlobStore` in the interfaces import line):

```python
def build_blob_store(settings: Settings | None = None) -> BlobStore:
    s = settings or get_settings()
    from providers.blobstore.local_disk import LocalDiskBlobStore

    return LocalDiskBlobStore(root=s.blob_store_root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_blobstore.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/config.py core/interfaces.py providers/blobstore core/registry.py tests/test_blobstore.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add BlobStore interface and LocalDiskBlobStore" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 9: Document registry (status lifecycle, tenant-scoped)

**Files:**
- Modify: `core/types.py`
- Modify: `core/interfaces.py`
- Modify: `core/config.py`
- Create: `providers/docstore/__init__.py`
- Create: `providers/docstore/memory.py`
- Create: `providers/docstore/postgres.py`
- Modify: `core/registry.py`
- Test: `tests/test_document_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DocumentStatus` (str Enum: `PROCESSING="processing"`,
  `READY="ready"`, `FAILED="failed"`); `DocumentRecord(document_id, tenant_id,
  filename, content_type, size_bytes, status, blob_key, error="", chunk_count=0)`;
  `DocumentRegistry` Protocol with `create(record) -> None`,
  `get(document_id, tenant_id) -> DocumentRecord | None`,
  `list(tenant_id) -> list[DocumentRecord]`,
  `set_status(document_id, tenant_id, status, *, error="", chunk_count=0) -> None`;
  `InMemoryDocumentRegistry`; `PostgresDocumentRegistry(dsn, table)`;
  `build_document_registry(settings) -> DocumentRegistry`. `get`/`set_status`
  are tenant-scoped (a mismatched tenant sees `None` / no-op).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_document_registry.py
from core.types import DocumentRecord, DocumentStatus
from providers.docstore.memory import InMemoryDocumentRegistry


def _rec(did="doc1", tenant="t1"):
    return DocumentRecord(document_id=did, tenant_id=tenant, filename="f.pdf",
                          content_type="application/pdf", size_bytes=10,
                          status=DocumentStatus.PROCESSING, blob_key=f"{tenant}/{did}")


def test_create_and_get():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    got = reg.get("doc1", "t1")
    assert got is not None and got.status == DocumentStatus.PROCESSING


def test_get_is_tenant_scoped():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    assert reg.get("doc1", "other") is None  # existence never leaks cross-tenant


def test_set_status_transitions():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    reg.set_status("doc1", "t1", DocumentStatus.READY, chunk_count=5)
    got = reg.get("doc1", "t1")
    assert got.status == DocumentStatus.READY and got.chunk_count == 5


def test_list_scoped_to_tenant():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec("d1", "t1"))
    reg.create(_rec("d2", "t2"))
    assert [r.document_id for r in reg.list("t1")] == ["d1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_document_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: providers.docstore`).

- [ ] **Step 3: Implement types, protocol, stores, config, builder**

In `core/types.py` add:

```python
class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    blob_key: str
    error: str = ""
    chunk_count: int = 0
```

In `core/interfaces.py` add (import `DocumentRecord`, `DocumentStatus`):

```python
@runtime_checkable
class DocumentRegistry(Protocol):
    def create(self, record: "DocumentRecord") -> None: ...
    def get(self, document_id: str, tenant_id: str) -> "DocumentRecord | None": ...
    def list(self, tenant_id: str) -> "list[DocumentRecord]": ...
    def set_status(self, document_id: str, tenant_id: str, status: "DocumentStatus",
                   *, error: str = "", chunk_count: int = 0) -> None: ...
```

In `core/config.py` add:

```python
    doc_registry_backend: Literal["memory", "postgres"] = "postgres"
    doc_registry_table: str = "documents"
```

Create `providers/docstore/__init__.py` (empty). Create `providers/docstore/memory.py`:

```python
from __future__ import annotations

from core.types import DocumentRecord, DocumentStatus


class InMemoryDocumentRegistry:
    def __init__(self) -> None:
        self._rows: dict[str, DocumentRecord] = {}

    def create(self, record: DocumentRecord) -> None:
        self._rows[record.document_id] = record

    def get(self, document_id: str, tenant_id: str) -> DocumentRecord | None:
        r = self._rows.get(document_id)
        return r if r and r.tenant_id == tenant_id else None

    def list(self, tenant_id: str) -> list[DocumentRecord]:
        return [r for r in self._rows.values() if r.tenant_id == tenant_id]

    def set_status(self, document_id: str, tenant_id: str, status: DocumentStatus,
                   *, error: str = "", chunk_count: int = 0) -> None:
        r = self.get(document_id, tenant_id)
        if r is None:
            return
        self._rows[document_id] = r.model_copy(
            update={"status": status, "error": error, "chunk_count": chunk_count})
```

Create `providers/docstore/postgres.py`:

```python
from __future__ import annotations

from core.types import DocumentRecord, DocumentStatus

_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    document_id  TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    status       TEXT NOT NULL,
    blob_key     TEXT NOT NULL,
    error        TEXT NOT NULL DEFAULT '',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_tenant_idx ON {table} (tenant_id);
"""


class PostgresDocumentRegistry:
    def __init__(self, dsn: str, table: str = "documents") -> None:
        self._dsn = dsn
        self._table = table
        self._ensure()

    def _conn(self):
        import psycopg  # local import
        return psycopg.connect(self._dsn)

    def _ensure(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(_DDL.format(table=self._table))

    def create(self, record: DocumentRecord) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table} (document_id, tenant_id, filename, "
                f"content_type, size_bytes, status, blob_key, error, chunk_count) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (document_id) DO NOTHING",
                [record.document_id, record.tenant_id, record.filename,
                 record.content_type, record.size_bytes, record.status.value,
                 record.blob_key, record.error, record.chunk_count])

    def get(self, document_id: str, tenant_id: str) -> DocumentRecord | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT document_id, tenant_id, filename, content_type, size_bytes, "
                f"status, blob_key, error, chunk_count FROM {self._table} "
                f"WHERE document_id=%s AND tenant_id=%s", [document_id, tenant_id])
            row = cur.fetchone()
        return self._row(row) if row else None

    def list(self, tenant_id: str) -> list[DocumentRecord]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT document_id, tenant_id, filename, content_type, size_bytes, "
                f"status, blob_key, error, chunk_count FROM {self._table} "
                f"WHERE tenant_id=%s ORDER BY created_at DESC", [tenant_id])
            rows = cur.fetchall()
        return [self._row(r) for r in rows]

    def set_status(self, document_id: str, tenant_id: str, status: DocumentStatus,
                   *, error: str = "", chunk_count: int = 0) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"UPDATE {self._table} SET status=%s, error=%s, chunk_count=%s, "
                f"updated_at=now() WHERE document_id=%s AND tenant_id=%s",
                [status.value, error, chunk_count, document_id, tenant_id])

    @staticmethod
    def _row(row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row[0], tenant_id=row[1], filename=row[2], content_type=row[3],
            size_bytes=row[4], status=DocumentStatus(row[5]), blob_key=row[6],
            error=row[7], chunk_count=row[8])
```

In `core/registry.py` add:

```python
def build_document_registry(settings: Settings | None = None):
    s = settings or get_settings()
    if s.doc_registry_backend == "memory":
        from providers.docstore.memory import InMemoryDocumentRegistry

        return InMemoryDocumentRegistry()
    from providers.docstore.postgres import PostgresDocumentRegistry

    return PostgresDocumentRegistry(s.pg_dsn, s.doc_registry_table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_document_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/types.py core/interfaces.py core/config.py providers/docstore core/registry.py tests/test_document_registry.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add document registry with tenant-scoped status lifecycle" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 10: Ingest worker (parse → PII → chunk → incremental ingest)

**Files:**
- Modify: `core/config.py`
- Create: `ingest/worker.py`
- Test: `tests/test_ingest_worker.py`

**Interfaces:**
- Consumes: `ParserRegistry` (Task 7), `BlobStore` (Task 8), `DocumentRegistry`
  (Task 9), `IncrementalIngestor` (Task 6), `build_pii_detector`,
  `ingest.run._apply_pii_ingest_policy`, `ingest.chunking.chunk_document`,
  `PIIAuditLog`, `Settings`.
- Produces: `IngestDeps` dataclass (registry, blobs, parsers, ingestor, settings);
  `run_ingest(deps: IngestDeps, document_id: str) -> None` — the pure worker body
  that moves a document `processing → ready` (or `failed`). `arq`'s task function
  `ingest_document(ctx, document_id)` wraps `run_ingest` with deps built from
  `ctx`. `WorkerSettings` (arq) with `redis_settings` from config.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_worker.py
from core.config import Settings
from core.types import DocumentRecord, DocumentStatus
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k):
        if k not in self.d: raise KeyError(k)
        return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def _deps(tmp_path):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    ingestor = IncrementalIngestor(FakeEmbedder(), InMemoryVectorStore(),
                                   BM25Retriever(), JsonlManifestStore(str(tmp_path)))
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    settings = Settings(pii_mode="keep")
    return IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=settings)


def _seed(deps, did="doc1", tenant="t1", body=b"alpha beta gamma"):
    deps.blobs.put(f"{tenant}/{did}", body)
    deps.registry.create(DocumentRecord(
        document_id=did, tenant_id=tenant, filename="f.txt", content_type="text/plain",
        size_bytes=len(body), status=DocumentStatus.PROCESSING, blob_key=f"{tenant}/{did}"))


def test_worker_moves_document_to_ready(tmp_path):
    deps = _deps(tmp_path)
    _seed(deps)
    run_ingest(deps, "doc1")
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.READY
    assert rec.chunk_count >= 1
    assert len(deps.ingestor._store.chunks) >= 1


def test_worker_marks_failed_on_parse_error(tmp_path):
    deps = _deps(tmp_path)
    _seed(deps, body=b"x")
    # Force a parser failure: unknown type slips past by mutating the record.
    deps.registry._rows["doc1"] = deps.registry._rows["doc1"].model_copy(
        update={"content_type": "application/x-nope"})
    run_ingest(deps, "doc1")
    rec = deps.registry.get("doc1", "t1")
    assert rec.status == DocumentStatus.FAILED
    assert rec.error
    assert len(deps.ingestor._store.chunks) == 0  # no partial index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest_worker.py -v`
Expected: FAIL (`ModuleNotFoundError: ingest.worker`).

- [ ] **Step 3: Implement the worker**

In `core/config.py` add:

```python
    redis_url: str = "redis://localhost:6379"
    redis_password: str = ""
    ingest_queue_name: str = "ingest"
```

Create `ingest/worker.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config import Settings
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
    # The worker trusts the registry row for identity; we look it up by id across
    # tenants via a privileged path — the record carries the tenant it was created under.
    rec = None
    for candidate in getattr(deps.registry, "_rows", {}).values() if hasattr(deps.registry, "_rows") else []:
        if candidate.document_id == document_id:
            rec = candidate
            break
    if rec is None and hasattr(deps.registry, "get_privileged"):
        rec = deps.registry.get_privileged(document_id)
    if rec is None:
        logger.error("ingest: document %s not found", document_id)
        return

    tenant_id = rec.tenant_id
    acl = ACLContext(tenant_id=tenant_id, acl_tags=())
    try:
        deps.parsers.guard_size(b"")  # size already checked at upload; no-op guard
        parser = deps.parsers.resolve(rec.content_type)
        raw = deps.blobs.get(rec.blob_key)
        docs = parser.parse(raw, rec.filename, rec.content_type,
                            doc_id=document_id, tenant_id=tenant_id, acl_tags=())
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


class WorkerSettings:
    """arq worker configuration (see `arq ingest.worker.WorkerSettings`)."""
    functions = [ingest_document]

    @staticmethod
    def redis_settings():
        from arq.connections import RedisSettings
        from core.config import get_settings
        s = get_settings()
        return RedisSettings.from_dsn(s.redis_url)
```

Add a privileged lookup to `providers/docstore/memory.py` and
`providers/docstore/postgres.py` so the worker can resolve a doc by id without a
tenant (the worker is trusted; the row carries the tenant):

```python
    # InMemoryDocumentRegistry
    def get_privileged(self, document_id: str):
        return self._rows.get(document_id)
```

```python
    # PostgresDocumentRegistry
    def get_privileged(self, document_id: str):
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT document_id, tenant_id, filename, content_type, size_bytes, "
                f"status, blob_key, error, chunk_count FROM {self._table} "
                f"WHERE document_id=%s", [document_id])
            row = cur.fetchone()
        return self._row(row) if row else None
```

Simplify `run_ingest` to use `get_privileged` (replace the `_rows` scan with):

```python
    rec = deps.registry.get_privileged(document_id)
    if rec is None:
        logger.error("ingest: document %s not found", document_id)
        return
```

Add `arq>=0.26` to the `app` extra in `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest_worker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/config.py ingest/worker.py providers/docstore tests/test_ingest_worker.py pyproject.toml
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add async ingest worker body and arq wiring" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 11: Upload + status API routes

**Files:**
- Create: `app/documents.py`
- Modify: `app/api.py`
- Test: `tests/test_documents_api.py`

**Interfaces:**
- Consumes: `require_principal` (from `app.auth`), `DocumentRegistry`,
  `BlobStore`, `ParserRegistry`, `DocumentRecord`, `DocumentStatus`, `Principal`.
- Produces: an `APIRouter` with `POST /documents` (multipart `file`),
  `GET /documents/{document_id}`, `GET /documents`. An enqueue seam
  `get_enqueuer()` (FastAPI dependency) returning an async callable
  `enqueue(document_id: str) -> None`; the default posts to arq, tests override it.
  Registry/blobs/parsers are also dependencies (`get_registry`, `get_blobs`,
  `get_parsers`) so tests inject in-memory fakes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_documents_api.py
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app import documents as docs_mod
from core.types import Principal
from ingest.parsers.base import ParserRegistry
from providers.docstore.memory import InMemoryDocumentRegistry


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k): return self.d[k]
    def delete(self, k): self.d.pop(k, None)


@pytest.fixture
def client():
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers

    async def fake_enqueue(document_id): enqueued.append(document_id)
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue

    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    c = TestClient(app)
    c.enqueued = enqueued
    c.registry = reg
    yield c
    app.dependency_overrides.clear()


def test_upload_returns_202_and_enqueues(client):
    r = client.post("/documents", files={"file": ("n.txt", b"hello", "text/plain")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing"
    assert client.enqueued == [body["document_id"]]


def test_upload_rejects_disallowed_type(client):
    r = client.post("/documents", files={"file": ("n.bin", b"x", "application/x-nope")})
    assert r.status_code == 415


def test_upload_rejects_oversize(client):
    big = b"x" * 2000  # parsers max_bytes=1000
    r = client.post("/documents", files={"file": ("n.txt", big, "text/plain")})
    assert r.status_code == 413


def test_get_status_is_tenant_scoped(client):
    r = client.post("/documents", files={"file": ("n.txt", b"hi", "text/plain")})
    did = r.json()["document_id"]
    # Same tenant sees it
    assert client.get(f"/documents/{did}").status_code == 200
    # Switch principal to another tenant -> 404
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="other")
    assert client.get(f"/documents/{did}").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_documents_api.py -v`
Expected: FAIL (`ModuleNotFoundError: app.documents`).

- [ ] **Step 3: Implement the router and mount it**

Create `app/documents.py`:

```python
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.auth import require_principal
from core.registry import (build_blob_store, build_document_registry,
                           build_parser_registry)
from core.types import DocumentRecord, DocumentStatus, Principal
from ingest.parsers.base import ParserError

router = APIRouter()

_registry = None
_blobs = None
_parsers = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = build_document_registry()
    return _registry


def get_blobs():
    global _blobs
    if _blobs is None:
        _blobs = build_blob_store()
    return _blobs


def get_parsers():
    global _parsers
    if _parsers is None:
        _parsers = build_parser_registry()
    return _parsers


def get_enqueuer():
    """Return an async enqueue(document_id) that posts an arq job. Overridable."""
    async def enqueue(document_id: str) -> None:
        from arq import create_pool
        from arq.connections import RedisSettings
        from core.config import get_settings
        s = get_settings()
        pool = await create_pool(RedisSettings.from_dsn(s.redis_url))
        try:
            await pool.enqueue_job("ingest_document", document_id)
        finally:
            await pool.close()
    return enqueue


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile,
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
    blobs=Depends(get_blobs),
    parsers=Depends(get_parsers),
    enqueue=Depends(get_enqueuer),
):
    content_type = file.content_type or "application/octet-stream"
    try:
        parsers.resolve(content_type)  # allowlist check (415 on reject)
    except ParserError:
        raise HTTPException(status_code=415, detail=f"unsupported type: {content_type}")

    raw = await file.read()
    try:
        parsers.guard_size(raw)  # size check (413 on reject)
    except ParserError:
        raise HTTPException(status_code=413, detail="file too large")

    document_id = str(uuid.uuid4())
    blob_key = f"{principal.tenant_id}/{document_id}"
    blobs.put(blob_key, raw)
    registry.create(DocumentRecord(
        document_id=document_id, tenant_id=principal.tenant_id,
        filename=file.filename or document_id, content_type=content_type,
        size_bytes=len(raw), status=DocumentStatus.PROCESSING, blob_key=blob_key))
    try:
        await enqueue(document_id)
    except Exception:
        raise HTTPException(status_code=503, detail="ingest queue unavailable")
    return {"document_id": document_id, "status": DocumentStatus.PROCESSING.value}


@router.get("/documents/{document_id}")
def get_document(document_id: str, principal: Principal = Depends(require_principal),
                 registry=Depends(get_registry)):
    rec = registry.get(document_id, principal.tenant_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="not found")
    return rec.model_dump()


@router.get("/documents")
def list_documents(principal: Principal = Depends(require_principal),
                   registry=Depends(get_registry)):
    return [r.model_dump() for r in registry.list(principal.tenant_id)]
```

In `app/api.py`, mount the router (after `app = FastAPI(...)`):

```python
from app.documents import router as documents_router

app.include_router(documents_router)
```

Add `python-multipart>=0.0.9` to the `app` extra in `pyproject.toml` (FastAPI
needs it for `UploadFile`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_documents_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/documents.py app/api.py pyproject.toml tests/test_documents_api.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(api): add async document upload and status endpoints" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 12: End-to-end — upload → ingest → query retrieves the doc

**Files:**
- Test: `tests/test_ingest_query_e2e.py`

**Interfaces:**
- Consumes: everything above (API router, `run_ingest`, `IncrementalIngestor`,
  `HybridRetriever`, fakes). No new production code — this task proves the seams
  fit together and guards against regressions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_query_e2e.py
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app import documents as docs_mod
from core.config import Settings
from core.types import ACLContext, Principal, Query
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore


class DictBlobs:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k): return self.d[k]
    def delete(self, k): self.d.pop(k, None)


def test_uploaded_document_is_retrievable(tmp_path):
    reg = InMemoryDocumentRegistry()
    blobs = DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=10000)
    store = InMemoryVectorStore()
    sparse = BM25Retriever()
    ingestor = IncrementalIngestor(FakeEmbedder(), store, sparse,
                                   JsonlManifestStore(str(tmp_path)))
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    async def fake_enqueue(document_id): enqueued.append(document_id)
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    try:
        client = TestClient(app)
        body = b"the quick brown fox jumps over the lazy dog"
        r = client.post("/documents", files={"file": ("f.txt", body, "text/plain")})
        did = r.json()["document_id"]

        # Run the worker synchronously against the SAME fakes.
        deps = IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                          ingestor=ingestor, settings=Settings(pii_mode="keep"))
        run_ingest(deps, did)
        assert client.get(f"/documents/{did}").json()["status"] == "ready"

        # Retrieve via the hybrid path over the same store+sparse.
        retriever = HybridRetriever(FakeEmbedder(), store, sparse, FakeReranker())
        hits = retriever.retrieve(Query(text="quick brown fox",
                                        acl=ACLContext(tenant_id="t1"), top_k=5, rerank_top_n=3))
        assert any(h.chunk.doc_id == did for h in hits)

        # Isolation: another tenant sees nothing.
        empty = retriever.retrieve(Query(text="quick brown fox",
                                         acl=ACLContext(tenant_id="t2"), top_k=5, rerank_top_n=3))
        assert empty == []
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest_query_e2e.py -v`
Expected: FAIL initially only if an earlier seam is missing; if all prior tasks are done it should pass. If it fails, fix the seam it exposes (do not weaken the test).

- [ ] **Step 3: (No new code expected)**

This task is integration proof. If it fails, the fix belongs in the task that owns
the broken seam — return there, add/adjust a unit test, fix, and re-run.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (all prior tests plus this one).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest_query_e2e.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "test(ingest): end-to-end upload-to-retrieval over uploaded document" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

## Post-plan wiring (not a code task, do before deploying)

- **Compose:** add an `arq ingest.worker.WorkerSettings` worker service and point
  `redis_url` at the existing Redis (`infra/docker-compose.yml`). This belongs to
  SP9 (deployability) but the worker command is: `uv run arq ingest.worker.WorkerSettings`.
- **SP9 body-size rule:** when SP9's 64 KiB global request-body limit lands, exempt
  `POST /documents` (its own `max_upload_bytes` governs it).
- **Config:** set `llm_base_url` + `llm_api_key` to your OpenAI-compatible router;
  leave the reranker as `local`.

## Self-Review

- **Spec coverage:** §4.1 router → Task 1; §4.2 parser → Task 7; §4.3 BlobStore →
  Task 8; §4.4 registry → Task 9; §4.5 SP8 engine → Tasks 2, 3, 6; §4.6 per-tenant
  BM25 → Tasks 4, 5; §4.7 arq worker → Task 10; §4.8 API routes → Task 11;
  §5 data flow → Tasks 10–12; §6 error handling (415/413/503/failed/tenant-scope) →
  Tasks 9, 10, 11; §7 testing → every task + Task 12. No uncovered requirement.
- **Deferred correctly:** semantic cache, per-doc/collection query filter, DELETE
  endpoint, and eval rework are out of scope per the spec (§2) and appear in no task.
- **Type consistency:** `IncrementalIngestor.ingest_document(tenant_id, doc_id, chunks, acl)`
  used identically in Tasks 6, 10, 12; `DocumentStatus` values `processing/ready/failed`
  consistent across Tasks 9–12; `run_ingest(deps, document_id)` signature consistent
  Tasks 10 & 12; parser `parse(...)` keyword args consistent Tasks 7, 10, 12.
