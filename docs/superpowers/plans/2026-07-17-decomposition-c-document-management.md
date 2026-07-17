# Decomposition C — Document Management & Collection-Scoped Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a tenant list and delete their uploaded documents, and scope a query to a collection of documents, all with structural tenant isolation.

**Architecture:** Add a first-class `collection_id` to documents/chunks (assigned at upload, carried into every store's index payload, filterable pre-similarity via the existing `retrieval/acl.py` builders). Add tenant-scoped registry `delete` + an `IncrementalIngestor.delete_document` orchestration, driven by a new async `delete_document` arq job (`DELETE` returns 202). Expose `GET /documents`, `DELETE /documents/{id}`, upload `collection_id`, and query `collection_id`.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, Qdrant, pgvector (psycopg), rank-bm25, arq, pytest.

## Global Constraints

- **Commit authorship:** every commit authored solely as `Shreytam Goyal <shreytamgoyal@gmail.com>`; NO Claude/AI attribution of any kind (no `Co-Authored-By`, no `Claude-Session`). Use the exact `git -c user.name=... -c user.email=... commit` form shown in each task.
- **Tenant isolation is fail-closed and structural.** `collection_id` is ALWAYS an additional `AND` under the tenant+ACL predicate in the same filter builder — it can only narrow within a tenant, never widen or cross tenants. Never post-filter after `top_k` (that would silently under-return); filter pre-similarity.
- **Identity from the verified token only** (`require_principal`) — never from request body/headers.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- **Test env:** `uv run --extra all pytest` (has fastapi + langfuse + pytest). Live store tests (`qdrant`/`pgvector`) are gated/skipped without backends — do not require them locally.
- **`collection_id` is an opaque token**: empty string = unassigned; validated `≤ 128 chars, no control characters`; NEVER used in a filesystem path (blob keys stay `sha256(tenant_id)/document_id`).

---

### Task 1: Data model — `collection_id` fields + `DELETING` status

**Files:**
- Modify: `core/types.py` (`Document`, `Chunk`, `Query`, `DocumentRecord`, `DocumentStatus`)
- Test: `tests/test_collection_model.py`

**Interfaces:**
- Produces: `Document.collection_id: str = ""`, `Chunk.collection_id: str = ""`, `Query.collection_id: str | None = None`, `DocumentRecord.collection_id: str = ""`, `DocumentStatus.DELETING = "deleting"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collection_model.py
from core.types import Chunk, Document, DocumentRecord, DocumentStatus, Query
from core.types import ACLContext


def test_collection_id_defaults_empty():
    assert Document(doc_id="d", text="x", tenant_id="t").collection_id == ""
    assert Chunk(chunk_id="c", doc_id="d", text="x", tenant_id="t").collection_id == ""
    assert DocumentRecord(document_id="d", tenant_id="t", filename="f", content_type="text/plain",
                          size_bytes=1, status=DocumentStatus.PROCESSING, blob_key="k").collection_id == ""


def test_query_collection_id_optional():
    q = Query(text="hi", acl=ACLContext(tenant_id="t"))
    assert q.collection_id is None
    assert Query(text="hi", acl=ACLContext(tenant_id="t"), collection_id="col").collection_id == "col"


def test_deleting_status_exists():
    assert DocumentStatus.DELETING.value == "deleting"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_collection_model.py -v`
Expected: FAIL (`AttributeError`/`ValidationError`: no `collection_id` / no `DELETING`).

- [ ] **Step 3: Implement**

In `core/types.py`, add `collection_id: str = ""` to `Document` and `Chunk` (next to `acl_tags`), add `collection_id: str = ""` to `DocumentRecord` (next to `blob_key`), add `collection_id: str | None = None` to `Query`, and add `DELETING = "deleting"` to `DocumentStatus`:

```python
class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_collection_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/types.py tests/test_collection_model.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add collection_id to document/chunk/query model and DELETING status"
```

---

### Task 2: ACL filter builders accept `collection_id`

**Files:**
- Modify: `retrieval/acl.py` (`qdrant_filter`, `pg_where`, `acl_predicate`)
- Test: `tests/test_acl_collection_filter.py`

**Interfaces:**
- Consumes: `Chunk.collection_id` (Task 1).
- Produces: `qdrant_filter(acl, *, collection_id: str | None = None)`, `pg_where(acl, *, collection_id: str | None = None)`, `acl_predicate(acl, *, collection_id: str | None = None)`. `None` = no collection filter (unchanged behaviour).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_acl_collection_filter.py
from core.types import ACLContext, Chunk
from retrieval.acl import acl_predicate, pg_where, qdrant_filter


def _chunk(collection_id):
    return Chunk(chunk_id="c", doc_id="d", text="x", tenant_id="t", collection_id=collection_id)


def test_acl_predicate_collection_filter():
    acl = ACLContext(tenant_id="t")
    pred = acl_predicate(acl, collection_id="A")
    assert pred(_chunk("A")) is True
    assert pred(_chunk("B")) is False
    # No collection filter => collection ignored
    assert acl_predicate(acl)(_chunk("B")) is True


def test_pg_where_appends_collection():
    frag, params = pg_where(ACLContext(tenant_id="t"), collection_id="A")
    assert "collection_id = %s" in frag
    assert params[-1] == "A"
    # None => no collection clause
    frag2, _ = pg_where(ACLContext(tenant_id="t"))
    assert "collection_id" not in frag2


def test_qdrant_filter_appends_collection():
    f = qdrant_filter(ACLContext(tenant_id="t"), collection_id="A")
    keys = [getattr(c, "key", None) for c in f.must]
    assert "collection_id" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_acl_collection_filter.py -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'collection_id'`).

- [ ] **Step 3: Implement**

In `retrieval/acl.py`:

```python
def qdrant_filter(acl: ACLContext, *, collection_id: str | None = None) -> qm.Filter:
    visibility_should: list[qm.Condition] = [
        qm.FieldCondition(key="acl_open", match=qm.MatchValue(value=True))
    ]
    if acl.acl_tags:
        visibility_should.append(
            qm.FieldCondition(key="acl_tags", match=qm.MatchAny(any=list(acl.acl_tags)))
        )
    must: list[qm.Condition] = [
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=acl.tenant_id)),
        qm.Filter(should=visibility_should),
    ]
    if collection_id is not None:
        must.append(
            qm.FieldCondition(key="collection_id", match=qm.MatchValue(value=collection_id))
        )
    return qm.Filter(must=must)


def pg_where(acl: ACLContext, *, collection_id: str | None = None) -> tuple[str, list]:
    fragment = "tenant_id = %s AND (cardinality(acl_tags)=0 OR acl_tags && %s)"
    params: list = [acl.tenant_id, list(acl.acl_tags)]
    if collection_id is not None:
        fragment += " AND collection_id = %s"
        params.append(collection_id)
    return fragment, params


def acl_predicate(acl: ACLContext, *, collection_id: str | None = None) -> Callable[[Chunk], bool]:
    if collection_id is None:
        return lambda chunk: acl.allows(chunk.acl)
    return lambda chunk: acl.allows(chunk.acl) and chunk.collection_id == collection_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_acl_collection_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/acl.py tests/test_acl_collection_filter.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(retrieval): thread optional collection_id through acl filter builders"
```

---

### Task 3: Offline stores honour `collection_id` (BM25 + tenant sparse + in-memory fake)

**Files:**
- Modify: `providers/sparse/bm25.py` (`BM25Retriever.search`), `providers/sparse/tenant_store.py` (`TenantSparseStore.search`), `tests/_fakes.py` (`InMemoryVectorStore.search`)
- Modify: `core/interfaces.py` (`VectorStore.search`, `SparseRetriever.search` Protocol signatures)
- Test: `tests/test_collection_scoped_offline.py`

**Interfaces:**
- Consumes: `acl_predicate(acl, *, collection_id=...)` (Task 2), `Chunk.collection_id` (Task 1).
- Produces: `BM25Retriever.search(query, top_k, acl, *, collection_id=None)`, `TenantSparseStore.search(query, top_k, acl, *, collection_id=None)`, `InMemoryVectorStore.search(embedding, top_k, acl, *, collection_id=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collection_scoped_offline.py
from core.types import ACLContext, Chunk
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _chunk(cid, col, text):
    return Chunk(chunk_id=cid, doc_id=cid, text=text, tenant_id="t", collection_id=col)


def test_bm25_collection_scoping():
    r = BM25Retriever()
    r.index([_chunk("a", "A", "alpha shared"), _chunk("b", "B", "beta shared")])
    acl = ACLContext(tenant_id="t")
    got = r.search("shared", 5, acl, collection_id="A")
    assert [s.chunk.chunk_id for s in got] == ["a"]
    assert {s.chunk.chunk_id for s in r.search("shared", 5, acl)} == {"a", "b"}


def test_inmemory_vector_collection_scoping():
    emb = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [_chunk("a", "A", "alpha shared"), _chunk("b", "B", "beta shared")]
    for c, v in zip(chunks, emb.embed_documents([c.text for c in chunks])):
        c.embedding = v
    store.upsert(chunks)
    acl = ACLContext(tenant_id="t")
    got = store.search(emb.embed_query("shared"), 5, acl, collection_id="A")
    assert [s.chunk.chunk_id for s in got] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_collection_scoped_offline.py -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'collection_id'`).

- [ ] **Step 3: Implement**

`providers/sparse/bm25.py` — change the `search` signature and predicate:

```python
    def search(self, query: str, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]:
        if acl.tenant_id not in self._indices:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        bm25, chunks = self._indices[acl.tenant_id]
        scores = bm25.get_scores(tokens)
        predicate = acl_predicate(acl, collection_id=collection_id)
        candidates: list[tuple[float, Chunk]] = [
            (float(scores[i]), chunk)
            for i, chunk in enumerate(chunks)
            if predicate(chunk)
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        results: list[ScoredChunk] = []
        for rank, (score, chunk) in enumerate(candidates[:top_k], start=1):
            results.append(ScoredChunk(chunk=chunk, score=score,
                                       source=RetrievalSource.SPARSE, rank=rank))
        return results
```

`providers/sparse/tenant_store.py` — pass it through:

```python
    def search(self, query: str, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]:
        return self._retriever(acl.tenant_id).search(query, top_k, acl, collection_id=collection_id)
```

`tests/_fakes.py` — `InMemoryVectorStore.search`:

```python
    def search(self, embedding, top_k, acl: ACLContext, *, collection_id: str | None = None):
        scored = []
        for c in self.chunks:
            if not acl.allows(c.acl):  # pre-similarity ACL gate
                continue
            if collection_id is not None and c.collection_id != collection_id:
                continue
            sim = sum(a * b for a, b in zip(embedding, c.embedding or []))
            scored.append(ScoredChunk(chunk=c, score=sim, source=RetrievalSource.DENSE))
        scored.sort(key=lambda s: s.score, reverse=True)
        for r, s in enumerate(scored[:top_k], 1):
            s.rank = r
        return scored[:top_k]
```

In `core/interfaces.py`, update the `VectorStore.search` and `SparseRetriever.search` Protocol methods to include `*, collection_id: str | None = None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_collection_scoped_offline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/sparse/bm25.py providers/sparse/tenant_store.py tests/_fakes.py core/interfaces.py tests/test_collection_scoped_offline.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(retrieval): collection_id scoping in bm25, tenant-sparse, and in-memory stores"
```

---

### Task 4: Qdrant store carries + filters `collection_id`

**Files:**
- Modify: `providers/vectorstores/qdrant_store.py` (`_payload_from_chunk`, `_chunk_from_payload`, `ensure_collection`, `search`)
- Test: `tests/test_stores_acl.py` (add a live-gated collection case alongside the existing Qdrant live tests)

**Interfaces:**
- Consumes: `qdrant_filter(acl, *, collection_id=...)` (Task 2).
- Produces: `QdrantVectorStore.search(embedding, top_k, acl, *, collection_id=None)`; payload key `collection_id`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stores_acl.py` inside the existing `TestQdrantVectorStoreLive` class (reuses its skip-guard + fixture):

```python
    def test_collection_scoping(self, store):
        from core.types import ACLContext
        a = self._chunk("cola", tenant="t", collection_id="A", text="shared alpha")
        b = self._chunk("colb", tenant="t", collection_id="B", text="shared beta")
        store.upsert([a, b])
        acl = ACLContext(tenant_id="t")
        hits = store.search(self.embed("shared"), 5, acl, collection_id="A")
        assert {h.chunk.chunk_id for h in hits} == {"cola"}
```

If the class lacks a `_chunk(... collection_id=...)`/`embed(...)` helper, add a minimal one that builds an embedded `Chunk` with the given `collection_id` (mirror the existing per-test chunk construction in that class).

- [ ] **Step 2: Run test to verify it fails (or skips without Qdrant)**

Run: `uv run --extra all pytest tests/test_stores_acl.py -k collection_scoping -v`
Expected: FAIL if Qdrant is up (`TypeError`/missing payload); SKIP if no live Qdrant. Both are acceptable — the store code change is still required.

- [ ] **Step 3: Implement**

In `providers/vectorstores/qdrant_store.py`:

```python
def _payload_from_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "tenant_id": chunk.tenant_id,
        "collection_id": chunk.collection_id,
        "acl_tags": list(chunk.acl_tags),
        "acl_open": not bool(chunk.acl_tags),
        # ... keep all existing keys (text, ordinal, title, source, contextual_prefix, metadata)
    }
```

In `_chunk_from_payload`, add `collection_id=payload.get("collection_id", "")` to the `Chunk(...)` call. In `ensure_collection`, add a payload index for `collection_id` (mirror the existing `tenant_id` `create_payload_index` call). Change `search` to accept `*, collection_id: str | None = None` and pass it: `query_filter=qdrant_filter(acl, collection_id=collection_id)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_stores_acl.py -k collection_scoping -v`
Expected: PASS with live Qdrant; SKIP otherwise. Also run the full `tests/test_stores_acl.py` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add providers/vectorstores/qdrant_store.py tests/test_stores_acl.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(vectorstore): store and filter collection_id in qdrant payload"
```

---

### Task 5: Remove pgvector — Qdrant is the sole vector store

> **Superseded (mid-execution decision):** the product stack standardized on Qdrant
> (vectors) + Redis (semantic cache). Rather than add `collection_id` to pgvector, the
> pgvector backend is **removed entirely**: delete `providers/vectorstores/pgvector_store.py`,
> drop the `"pgvector"` option from `core/config.py`'s `vector_store` Literal, remove the
> `PgVectorStore` branch/import in `core/registry.py`, delete `pg_where` from
> `retrieval/acl.py` (and its Task-2 test), and remove all pgvector-only tests. Keep the
> Postgres *document registry* and `pg_dsn` (a separate subsystem). The original
> pgvector-collection task text below is retained for history but does NOT apply.

**Files (original — superseded):**
- Modify: `providers/vectorstores/pgvector_store.py` (`ensure_collection` DDL, `upsert`, `search`)
- Test: `tests/test_stores_acl.py` (live-gated pgvector collection case, mirroring Task 4)

**Interfaces:**
- Consumes: `pg_where(acl, *, collection_id=...)` (Task 2).
- Produces: `PgVectorStore.search(embedding, top_k, acl, *, collection_id=None)`; `collection_id TEXT` column.

- [ ] **Step 1: Write the failing test**

Add a `test_collection_scoping` to the existing `TestPgVectorStoreLive` class, structured exactly like the Qdrant one in Task 4 (upsert two chunks in collections A and B, assert a scoped search returns only A).

- [ ] **Step 2: Run test to verify it fails (or skips without Postgres)**

Run: `uv run --extra all pytest tests/test_stores_acl.py -k "collection_scoping and Pg" -v`
Expected: FAIL with live Postgres; SKIP otherwise.

- [ ] **Step 3: Implement**

In `pgvector_store.py`:
- `ensure_collection`: add `collection_id TEXT NOT NULL DEFAULT ''` to the `CREATE TABLE` column list, and a `CREATE INDEX IF NOT EXISTS {table}_collection_idx ON {table} (collection_id)`.
- `upsert`: add `collection_id` to the INSERT column list, the `VALUES` placeholder count (one more `%s`), the `ON CONFLICT ... DO UPDATE SET collection_id = EXCLUDED.collection_id`, and the params tuple (`chunk.collection_id`).
- `search`: change signature to `search(self, embedding, top_k, acl, *, collection_id: str | None = None)`; call `where_sql, where_params = pg_where(acl, collection_id=collection_id)`. Add `collection_id` to the `SELECT` column list and reconstruct it on the `Chunk(...)` (unpack an extra column). Keep placeholder ordering correct (SELECT-distance embedding, WHERE params, ORDER-BY embedding, LIMIT).

> Note: this changes the table schema. In dev the table is created fresh by `ensure_collection`; an existing table needs `ALTER TABLE ... ADD COLUMN collection_id TEXT NOT NULL DEFAULT ''` (document this in the commit body — no migration framework in the repo).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_stores_acl.py -k "collection_scoping and Pg" -v`
Expected: PASS with live Postgres; SKIP otherwise.

- [ ] **Step 5: Commit**

```bash
git add providers/vectorstores/pgvector_store.py tests/test_stores_acl.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(vectorstore): store and filter collection_id in pgvector (schema + where)"
```

---

### Task 6: Registry `delete` + `collection_id` (memory + postgres)

**Files:**
- Modify: `providers/docstore/memory.py`, `providers/docstore/postgres.py`
- Test: `tests/test_document_registry.py` (extend)

**Interfaces:**
- Consumes: `DocumentRecord.collection_id` (Task 1).
- Produces: `InMemoryDocumentRegistry.delete(document_id, tenant_id) -> None`, `PostgresDocumentRegistry.delete(document_id, tenant_id) -> None`; both registries round-trip `collection_id`; `list(tenant_id)` returns records carrying `collection_id`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_document_registry.py
from core.types import DocumentRecord, DocumentStatus
from providers.docstore.memory import InMemoryDocumentRegistry


def _rec(did, tenant="t", col=""):
    return DocumentRecord(document_id=did, tenant_id=tenant, filename="f",
                          content_type="text/plain", size_bytes=1,
                          status=DocumentStatus.PROCESSING, blob_key="k", collection_id=col)


def test_delete_is_tenant_scoped_and_idempotent():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec("d1", "t1", col="A"))
    # cross-tenant delete is a no-op
    reg.delete("d1", "other")
    assert reg.get("d1", "t1") is not None
    assert reg.get("d1", "t1").collection_id == "A"
    reg.delete("d1", "t1")
    assert reg.get("d1", "t1") is None
    reg.delete("d1", "t1")  # idempotent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_document_registry.py::test_delete_is_tenant_scoped_and_idempotent -v`
Expected: FAIL (`AttributeError: ... no attribute 'delete'`).

- [ ] **Step 3: Implement**

`providers/docstore/memory.py` — add:

```python
    def delete(self, document_id: str, tenant_id: str) -> None:
        r = self._rows.get(document_id)
        if r is not None and r.tenant_id == tenant_id:
            self._rows.pop(document_id, None)
```

`providers/docstore/postgres.py`:
- `_DDL`: add `collection_id TEXT NOT NULL DEFAULT ''` to the `CREATE TABLE`.
- `create`: add `collection_id` to the INSERT column list, one more `%s`, and `record.collection_id` in the params.
- every `SELECT` column list (`get`, `get_privileged`, `list`): add `collection_id`; `_row`: add `collection_id=row[9]` (shift indexing as needed).
- add:

```python
    def delete(self, document_id: str, tenant_id: str) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE document_id=%s AND tenant_id=%s",
                        [document_id, tenant_id])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_document_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/docstore/memory.py providers/docstore/postgres.py tests/test_document_registry.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): tenant-scoped registry delete + collection_id column"
```

---

### Task 7: `IncrementalIngestor.delete_document`

**Files:**
- Modify: `ingest/incremental.py`
- Test: `tests/test_incremental_ingestor.py` (extend)

**Interfaces:**
- Consumes: `manifest.load(tenant_id, doc_id)`, `manifest.delete(tenant_id, doc_id)`, `store.delete(chunk_ids, acl)`, `sparse.delete(chunk_ids, acl)` (all exist).
- Produces: `IncrementalIngestor.delete_document(tenant_id, doc_id, acl) -> int` (returns #chunks removed).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_incremental_ingestor.py
from core.types import ACLContext, Chunk
from ingest.incremental import IncrementalIngestor
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _c(doc, ordinal, text="x"):
    return Chunk(chunk_id=f"{doc}::{ordinal}", doc_id=doc, text=text, tenant_id="t")


def test_delete_document_removes_only_that_doc(tmp_path):
    store, sparse = InMemoryVectorStore(), BM25Retriever()
    ing = IncrementalIngestor(FakeEmbedder(), store, sparse, JsonlManifestStore(str(tmp_path)))
    acl = ACLContext(tenant_id="t")
    ing.ingest_document("t", "d1", [_c("d1", 0), _c("d1", 1)], acl)
    ing.ingest_document("t", "d2", [_c("d2", 0)], acl)

    n = ing.delete_document("t", "d2", acl)
    assert n == 1
    assert {c.doc_id for c in store.chunks} == {"d1"}
    # idempotent: deleting an absent doc is a clean no-op
    assert ing.delete_document("t", "gone", acl) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_incremental_ingestor.py::test_delete_document_removes_only_that_doc -v`
Expected: FAIL (`AttributeError: ... no attribute 'delete_document'`).

- [ ] **Step 3: Implement**

Add to `IncrementalIngestor`:

```python
    def delete_document(self, tenant_id: str, doc_id: str, acl: ACLContext) -> int:
        old = self._manifest.load(tenant_id, doc_id)
        if old is None:
            return 0
        chunk_ids = list(old.chunks.keys())
        if chunk_ids:
            self._store.delete(chunk_ids, acl)
            self._sparse.delete(chunk_ids, acl)
        self._manifest.delete(tenant_id, doc_id)
        return len(chunk_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_incremental_ingestor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/incremental.py tests/test_incremental_ingestor.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): IncrementalIngestor.delete_document purges chunks + manifest"
```

---

### Task 8: Chunking copies `collection_id` doc → chunk

**Files:**
- Modify: `ingest/chunking.py` (the `Chunk(...)` construction)
- Test: `tests/test_chunking_collection.py`

**Interfaces:**
- Consumes: `Document.collection_id` (Task 1).
- Produces: chunks whose `collection_id` equals their source `Document.collection_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking_collection.py
from core.types import Document
from ingest.chunking import chunk_document


def test_chunks_inherit_collection_id():
    doc = Document(doc_id="d", text="alpha beta gamma " * 20, tenant_id="t", collection_id="A")
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.collection_id == "A" for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_chunking_collection.py -v`
Expected: FAIL (`collection_id == ""` not `"A"`).

- [ ] **Step 3: Implement**

In `ingest/chunking.py`, add `collection_id=doc.collection_id` to the `Chunk(...)` constructor (next to `acl_tags=doc.acl_tags`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_chunking_collection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/chunking.py tests/test_chunking_collection.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): propagate collection_id from document to chunks"
```

---

### Task 9: `HybridRetriever` threads `Query.collection_id`

**Files:**
- Modify: `retrieval/hybrid.py` (`retrieve`)
- Test: `tests/test_hybrid_collection.py`

**Interfaces:**
- Consumes: `Query.collection_id` (Task 1), `search(..., collection_id=...)` (Tasks 3–5).
- Produces: `HybridRetriever.retrieve` passes `query.collection_id` into both dense and sparse `search`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_collection.py
from core.types import ACLContext, Chunk, Query
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore


def _c(cid, col, text):
    return Chunk(chunk_id=cid, doc_id=cid, text=text, tenant_id="t", collection_id=col)


def test_retrieve_scopes_to_collection():
    emb, store, sparse = FakeEmbedder(), InMemoryVectorStore(), BM25Retriever()
    chunks = [_c("a", "A", "shared alpha"), _c("b", "B", "shared beta")]
    for c, v in zip(chunks, emb.embed_documents([c.text for c in chunks])):
        c.embedding = v
    store.upsert(chunks)
    sparse.index(chunks)
    r = HybridRetriever(emb, store, sparse, FakeReranker())
    hits = r.retrieve(Query(text="shared", acl=ACLContext(tenant_id="t"),
                            top_k=5, rerank_top_n=3, collection_id="A"))
    assert {h.chunk.chunk_id for h in hits} == {"a"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_hybrid_collection.py -v`
Expected: FAIL (returns both `a` and `b` — filter not applied).

- [ ] **Step 3: Implement**

In `retrieval/hybrid.py` `retrieve`:

```python
    def retrieve(self, query: Query) -> list[ScoredChunk]:
        qvec = self.embedder.embed_query(query.text)
        dense = self.vector_store.search(qvec, query.top_k, query.acl,
                                         collection_id=query.collection_id)
        sparse = self.sparse.search(query.text, query.top_k, query.acl,
                                    collection_id=query.collection_id)
        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k)
        if not fused:
            return []
        window = fused[: self.fuse_window]
        return self.reranker.rerank(query.text, window, query.rerank_top_n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_hybrid_collection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/hybrid.py tests/test_hybrid_collection.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(retrieval): HybridRetriever honours Query.collection_id on both legs"
```

---

### Task 10: Pipeline threads `collection_id` into `Query`

**Files:**
- Modify: `core/pipeline.py` (`answer`, `run`)
- Test: `tests/test_pipeline_collection.py`

**Interfaces:**
- Consumes: `Query.collection_id` (Task 1), `HybridRetriever` scoping (Task 9).
- Produces: `RAGPipeline.answer(question, acl=None, *, collection_id=None)` and `RAGPipeline.run(question, acl=None, *, collection_id=None)`; the built `Query` carries `collection_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_collection.py
from core.types import ACLContext, Query


class _Recorder:
    def __init__(self): self.q = None
    def retrieve(self, query: Query):
        self.q = query
        return []


def test_pipeline_passes_collection_id(monkeypatch):
    from core import pipeline as pipe
    # Build a minimal pipeline via the public build() with guardrails off, then
    # swap in a recording retriever. (If build() needs backends, construct
    # RAGPipeline directly with the fakes used in tests/test_sp4_pipeline_wire.py.)
    p = pipe.build(version="fast", corpus=None, enable_guardrails=False)
    rec = _Recorder()
    p.retriever = rec
    p.run("hello", ACLContext(tenant_id="t"), collection_id="A")
    assert rec.q is not None and rec.q.collection_id == "A"
```

> If `build(version="fast", ...)` is unavailable offline, mirror the pipeline construction already used in `tests/test_sp4_pipeline_wire.py` to assemble `RAGPipeline` from fakes, then assert the same.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_pipeline_collection.py -v`
Expected: FAIL (`TypeError: unexpected keyword 'collection_id'`).

- [ ] **Step 3: Implement**

In `core/pipeline.py`:
- `answer` signature → `def answer(self, question: str, acl: ACLContext | None = None, *, collection_id: str | None = None) -> Answer:` and set `collection_id=collection_id` in the `Query(...)` construction (line ~127).
- `run` signature → `def run(self, question: str, acl: ACLContext | None = None, *, collection_id: str | None = None) -> dict[str, Any]:` and forward: `ans = self.answer(question, acl, collection_id=collection_id)` (match however `run` currently delegates to `answer`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_pipeline_collection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/test_pipeline_collection.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pipeline): thread collection_id from run/answer into Query"
```

---

### Task 11: Worker — stamp `collection_id` + `run_delete` + delete task

**Files:**
- Modify: `ingest/worker.py` (`run_ingest`, add `run_delete`, add `delete_document` task, `WorkerSettings.functions`)
- Test: `tests/test_ingest_worker.py` (extend), `tests/test_ingest_delete_worker.py`

**Interfaces:**
- Consumes: `IncrementalIngestor.delete_document` (Task 7), `registry.delete` (Task 6), `blobs.delete`, `registry.get_privileged`, `DocumentStatus.DELETING/FAILED`.
- Produces: `run_delete(deps: IngestDeps, document_id: str) -> None`; async `delete_document(ctx, document_id)` arq task; `run_ingest` stamps `rec.collection_id` onto parsed docs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_delete_worker.py
from core.types import DocumentRecord, DocumentStatus
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_delete, run_ingest
from providers.docstore.memory import InMemoryDocumentRegistry
from providers.manifest.jsonl_store import JsonlManifestStore
from providers.sparse.bm25 import BM25Retriever
from core.config import Settings
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
    return IngestDeps(registry=reg, blobs=blobs, parsers=parsers,
                      ingestor=ingestor, settings=Settings(pii_mode="keep"))


def test_run_ingest_stamps_collection_id(tmp_path):
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta gamma")
    deps.registry.create(DocumentRecord(document_id="d1", tenant_id="t", filename="f.txt",
        content_type="text/plain", size_bytes=15, status=DocumentStatus.PROCESSING,
        blob_key="t/d1", collection_id="A"))
    run_ingest(deps, "d1")
    assert deps.ingestor._store.chunks
    assert all(c.collection_id == "A" for c in deps.ingestor._store.chunks)


def test_run_delete_purges_everything(tmp_path):
    deps = _deps(tmp_path)
    deps.blobs.put("t/d1", b"alpha beta gamma")
    deps.registry.create(DocumentRecord(document_id="d1", tenant_id="t", filename="f.txt",
        content_type="text/plain", size_bytes=15, status=DocumentStatus.PROCESSING,
        blob_key="t/d1", collection_id="A"))
    run_ingest(deps, "d1")
    run_delete(deps, "d1")
    assert deps.registry.get_privileged("d1") is None      # row gone (last)
    assert "t/d1" not in deps.blobs.d                        # blob gone
    assert deps.ingestor._store.chunks == []                # chunks gone
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra all pytest tests/test_ingest_delete_worker.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_delete'`, and collection_id stamping absent).

- [ ] **Step 3: Implement**

In `ingest/worker.py`:

Stamp collection_id in `run_ingest` — after `docs = parser.parse(...)`, before PII:

```python
        docs = parser.parse(raw, rec.filename, rec.content_type,
                            doc_id=document_id, tenant_id=tenant_id, acl_tags=())
        if rec.collection_id:
            docs = [d.model_copy(update={"collection_id": rec.collection_id}) for d in docs]
```

Add the delete body + task:

```python
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


async def delete_document(ctx, document_id: str) -> None:
    """arq task entrypoint for async document deletion."""
    deps = ctx.get("deps")
    if deps is None:
        from core.config import get_settings
        deps = _build_deps(get_settings())
        ctx["deps"] = deps
    run_delete(deps, document_id)
```

Add `delete_document` to `WorkerSettings.functions`:

```python
    functions = [ingest_document, delete_document]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra all pytest tests/test_ingest_delete_worker.py tests/test_ingest_worker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/worker.py tests/test_ingest_delete_worker.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): worker stamps collection_id and adds async run_delete task"
```

---

### Task 12: Upload accepts `collection_id` + `GET /documents` list

**Files:**
- Modify: `app/documents.py` (upload handler + new list route + `collection_id` validation)
- Test: `tests/test_documents_api.py` (extend)

**Interfaces:**
- Consumes: `registry.list(tenant)`, `DocumentRecord.collection_id`.
- Produces: `POST /documents` optional `collection_id` form field; `GET /documents` (optional `?collection_id=`); `_validate_collection_id(value) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_documents_api.py
def test_upload_stores_collection_id(client):
    r = client.post("/documents",
                    data={"collection_id": "projX"},
                    files={"file": ("n.txt", b"hi", "text/plain")})
    assert r.status_code == 202
    did = r.json()["document_id"]
    assert client.registry.get(did, "t1").collection_id == "projX"


def test_list_documents_tenant_scoped_and_filterable(client):
    a = client.post("/documents", data={"collection_id": "A"},
                    files={"file": ("a.txt", b"a", "text/plain")}).json()["document_id"]
    client.post("/documents", data={"collection_id": "B"},
                files={"file": ("b.txt", b"b", "text/plain")})
    all_docs = client.get("/documents").json()
    assert {d["document_id"] for d in all_docs} >= {a}
    only_a = client.get("/documents", params={"collection_id": "A"}).json()
    assert [d["document_id"] for d in only_a] == [a]
    assert all(d["collection_id"] == "A" for d in only_a)


def test_upload_rejects_bad_collection_id(client):
    r = client.post("/documents", data={"collection_id": "x" * 200},
                    files={"file": ("n.txt", b"hi", "text/plain")})
    assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_documents_api.py -k "collection or list_documents" -v`
Expected: FAIL (form field ignored; no `GET /documents` route).

- [ ] **Step 3: Implement**

In `app/documents.py`:

```python
from fastapi import Form, Query as QueryParam

_MAX_COLLECTION_ID = 128


def _validate_collection_id(value: str) -> str:
    if len(value) > _MAX_COLLECTION_ID or any(ord(ch) < 32 for ch in value):
        raise HTTPException(status_code=422, detail="invalid collection_id")
    return value
```

Add `collection_id: str = Form("")` to `upload_document`, validate it (`collection_id = _validate_collection_id(collection_id)`), and pass `collection_id=collection_id` into the `DocumentRecord(...)`.

Add the list route:

```python
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
        {"document_id": r.document_id, "filename": r.filename,
         "content_type": r.content_type, "size_bytes": r.size_bytes,
         "status": r.status.value, "chunk_count": r.chunk_count,
         "collection_id": r.collection_id, "error": r.error}
        for r in records
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_documents_api.py -v`
Expected: PASS (all existing + new)

- [ ] **Step 5: Commit**

```bash
git add app/documents.py tests/test_documents_api.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(api): accept collection_id on upload and add GET /documents list"
```

---

### Task 13: `DELETE /documents/{id}` (async) + enqueuer generalization

**Files:**
- Modify: `app/documents.py` (generalize enqueuer, add DELETE route)
- Test: `tests/test_documents_api.py` (extend)

**Interfaces:**
- Consumes: `registry.get`, `registry.set_status`, `DocumentStatus.DELETING`.
- Produces: enqueuer signature `enqueue(document_id, action="ingest")`; `DELETE /documents/{id}` → 202 / 404.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_documents_api.py
def test_delete_marks_deleting_and_enqueues(client):
    did = client.post("/documents", files={"file": ("n.txt", b"hi", "text/plain")}).json()["document_id"]
    client.enqueued.clear()
    r = client.delete(f"/documents/{did}")
    assert r.status_code == 202
    assert client.registry.get(did, "t1").status.value == "deleting"
    assert client.enqueued == [(did, "delete")]


def test_delete_cross_tenant_is_404(client):
    did = client.post("/documents", files={"file": ("n.txt", b"hi", "text/plain")}).json()["document_id"]
    from app.auth import require_principal
    from core.types import Principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="other")
    assert client.delete(f"/documents/{did}").status_code == 404
```

Update the `client` fixture's fake enqueuer to record the action:

```python
    async def fake_enqueue(document_id, action="ingest"):
        enqueued.append((document_id, action))
```

(and update the existing upload assertion `client.enqueued == [body["document_id"]]` to `client.enqueued == [(body["document_id"], "ingest")]`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_documents_api.py -k delete -v`
Expected: FAIL (no DELETE route).

- [ ] **Step 3: Implement**

In `app/documents.py`, generalize the real enqueuer:

```python
_ACTION_TO_FN = {"ingest": "ingest_document", "delete": "delete_document"}


def _build_arq_enqueuer() -> Callable[..., Awaitable[None]]:
    async def enqueue(document_id: str, action: str = "ingest") -> None:
        global _pool
        if _pool is None:
            from arq import create_pool
            from ingest.worker import WorkerSettings
            _pool = await create_pool(WorkerSettings.redis_settings())
        await _pool.enqueue_job(_ACTION_TO_FN[action], document_id)
    return enqueue
```

Update `upload_document` to call `await enqueue(document_id)` (default action unchanged). Add the DELETE route:

```python
@router.delete("/{document_id}", status_code=202)
async def delete_document(
    document_id: str,
    principal: Principal = Depends(require_principal),
    registry=Depends(get_registry),
    enqueue: Callable[..., Awaitable[None]] = Depends(get_enqueuer),
):
    record = registry.get(document_id, principal.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    registry.set_status(document_id, principal.tenant_id, DocumentStatus.DELETING)
    await enqueue(document_id, "delete")
    return {"document_id": document_id, "status": DocumentStatus.DELETING.value}
```

(Add `DocumentStatus` to the existing `core.types` import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_documents_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/documents.py tests/test_documents_api.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(api): async DELETE /documents/{id} + generalized ingest/delete enqueuer"
```

---

### Task 14: `POST /query` accepts `collection_id`

**Files:**
- Modify: `app/api.py` (`QueryRequest`, `query` handler)
- Test: `tests/test_query_collection_api.py`

**Interfaces:**
- Consumes: `pipeline.run(question, acl, *, collection_id=...)` (Task 10), `_validate` bound.
- Produces: `QueryRequest.collection_id: str | None`; handler forwards it to `pipeline.run`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_collection_api.py
from fastapi.testclient import TestClient
from app.api import app, get_pipeline
from app.auth import require_principal
from core.types import Principal


class _Pipe:
    def __init__(self): self.seen = None
    def run(self, question, acl, *, collection_id=None):
        self.seen = collection_id
        return {"answer": "ok", "citations": [], "retrieved_ids": [], "usage": {}, "refused": False}


def test_query_forwards_collection_id():
    pipe = _Pipe()
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")
    try:
        c = TestClient(app)
        r = c.post("/query", json={"question": "hi", "collection_id": "A"})
        assert r.status_code == 200
        assert pipe.seen == "A"
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra all pytest tests/test_query_collection_api.py -v`
Expected: FAIL (`collection_id` ignored → `pipe.seen is None`).

- [ ] **Step 3: Implement**

In `app/api.py`:
- `QueryRequest`: add `collection_id: str | None = Field(default=None, max_length=128)`.
- `query` handler: change the pipeline call to `result = pipeline.run(body.question, acl, collection_id=body.collection_id)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra all pytest tests/test_query_collection_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_query_collection_api.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(api): POST /query accepts optional collection_id"
```

---

### Task 15: End-to-end — collections scope retrieval; delete purges

**Files:**
- Test: `tests/test_collection_e2e.py`

**Interfaces:**
- Consumes: everything above. No new production code — integration proof + regression guard.

- [ ] **Step 1: Write the test**

```python
# tests/test_collection_e2e.py
from fastapi.testclient import TestClient

from app import documents as docs_mod
from app.api import app
from core.config import Settings
from core.types import ACLContext, Principal, Query
from ingest.incremental import IncrementalIngestor
from ingest.parsers.base import ParserRegistry
from ingest.worker import IngestDeps, run_delete, run_ingest
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


def test_collection_scoping_and_delete(tmp_path):
    reg, blobs = InMemoryDocumentRegistry(), DictBlobs()
    parsers = ParserRegistry(allowed_types={"text/plain"}, max_bytes=10000)
    store, sparse = InMemoryVectorStore(), BM25Retriever()
    ingestor = IncrementalIngestor(FakeEmbedder(), store, sparse, JsonlManifestStore(str(tmp_path)))
    enqueued = []

    app.dependency_overrides[docs_mod.get_registry] = lambda: reg
    app.dependency_overrides[docs_mod.get_blobs] = lambda: blobs
    app.dependency_overrides[docs_mod.get_parsers] = lambda: parsers
    async def fake_enqueue(document_id, action="ingest"): enqueued.append((document_id, action))
    app.dependency_overrides[docs_mod.get_enqueuer] = lambda: fake_enqueue
    from app.auth import require_principal
    app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="t1")

    deps = IngestDeps(registry=reg, blobs=blobs, parsers=parsers, ingestor=ingestor,
                      settings=Settings(pii_mode="keep"))
    try:
        client = TestClient(app)
        a = client.post("/documents", data={"collection_id": "A"},
                        files={"file": ("a.txt", b"the quick brown fox", "text/plain")}).json()["document_id"]
        b = client.post("/documents", data={"collection_id": "B"},
                        files={"file": ("b.txt", b"the quick brown fox", "text/plain")}).json()["document_id"]
        run_ingest(deps, a)
        run_ingest(deps, b)

        retriever = HybridRetriever(FakeEmbedder(), store, sparse, FakeReranker())
        hits = retriever.retrieve(Query(text="quick brown fox", acl=ACLContext(tenant_id="t1"),
                                        top_k=10, rerank_top_n=5, collection_id="A"))
        assert {h.chunk.doc_id for h in hits} == {a}

        # Delete A, then it disappears from retrieval and from the list.
        assert client.delete(f"/documents/{a}").status_code == 202
        run_delete(deps, a)
        hits2 = retriever.retrieve(Query(text="quick brown fox", acl=ACLContext(tenant_id="t1"),
                                         top_k=10, rerank_top_n=5, collection_id="A"))
        assert hits2 == []
        listed = client.get("/documents").json()
        assert a not in {d["document_id"] for d in listed}
        assert b in {d["document_id"] for d in listed}
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the test**

Run: `uv run --extra all pytest tests/test_collection_e2e.py -v`
Expected: PASS. If it fails, the fix belongs in the task owning the broken seam — return there, adjust a unit test, fix, re-run.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra all pytest -q`
Expected: PASS (all prior tests plus these).

- [ ] **Step 4: Commit**

```bash
git add tests/test_collection_e2e.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "test(ingest): end-to-end collection scoping + async delete"
```

---

## Post-plan wiring (not a code task)

- `docker compose` worker already runs `WorkerSettings.functions`, which now includes `delete_document` — no compose change needed; the same worker drains both queues.
- If a pgvector table pre-exists in a dev database, run `ALTER TABLE {pg_table} ADD COLUMN IF NOT EXISTS collection_id TEXT NOT NULL DEFAULT ''` once (no migration framework in the repo). Fresh tables get it from `ensure_collection`.

## Self-Review

- **Spec coverage:** §4.1 model → Task 1; §4.2 filters → Task 2; §4.3 stores → Tasks 3 (offline) + 4 (qdrant) + 5 (pgvector); §4.4 registry+ingestor → Tasks 6, 7; §4.5 worker → Task 11; §4.6 enqueuer → Task 13; §4.7 API (upload/list/delete/query) → Tasks 12, 13, 14; §4.8 Query+retriever → Tasks 1, 9; pipeline threading → Task 10; §5 data flow → Tasks 11, 9, 15; §6 error handling (404/202/422, push-down) → Tasks 12, 13, 14; §7 testing → every task + Task 15. No uncovered requirement.
- **Type consistency:** `search(..., *, collection_id: str | None = None)` identical across Tasks 3–5, 9; `acl_predicate/qdrant_filter/pg_where(..., *, collection_id=None)` identical Tasks 2–5; `delete_document(tenant_id, doc_id, acl) -> int` (ingestor, Task 7) vs `delete_document(ctx, document_id)` (arq task, Task 11) — distinct names in distinct namespaces, intentional; `run_delete(deps, document_id)` consistent Tasks 11, 15; `enqueue(document_id, action="ingest")` consistent Tasks 11, 13, 15; `registry.delete(document_id, tenant_id)` consistent Tasks 6, 11.
- **Placeholder scan:** none — every code step is concrete. The one `> Note` annotation (Task 10 pipeline-build fallback for offline construction) is an explicit implementer instruction, not an unresolved TBD.
