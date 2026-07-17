# Decomposition C — Document Management & Collection-Scoped Retrieval — Design

_Date: 2026-07-17_

## 1. Motivation

The product-ingestion path (decomposition A + B, shipped in `2026-07-17-product-ingestion-api-design.md`)
lets a tenant **upload** a document over the API and **query** it once `ready`. It has no way
to **manage** those documents afterward, and every query searches the tenant's *entire*
corpus. Two gaps remain from that spec's explicit "out of scope":

1. **Document management** — a tenant cannot enumerate what they have uploaded
   (`GET /documents`) or remove one (`DELETE /documents/{id}` — purge dense vectors + BM25 +
   manifest + raw blob + registry row).
2. **Scoped retrieval** — a query cannot be narrowed to a subset of the tenant's documents.
   A tenant that has uploaded many unrelated documents wants "answer using *these* documents,"
   not "answer using everything I've ever uploaded."

This spec closes both, and introduces the grouping primitive the scoped query needs.

## 2. Scope

### In scope (this spec)
- A first-class **`collection_id`** on documents and chunks: assigned at upload, carried
  through the worker into every chunk's index payload, filterable at query time.
- **`GET /documents`** — tenant-scoped list of a caller's documents, optionally filtered by
  `collection_id`.
- **`DELETE /documents/{id}`** — **asynchronous** purge: mark `deleting`, enqueue a delete
  job, return `202`; a worker removes the document's chunks (dense + sparse), manifest, raw
  blob, and finally the registry row.
- **Collection-scoped query** — `POST /query` accepts an optional `collection_id` that
  narrows retrieval to that collection, *always* ANDed under the existing tenant + ACL filter.
- The **storage-layer changes** this requires: `collection_id` in each store's filter surface
  (Qdrant payload + index, pgvector column, BM25 / in-memory Python filter), a tenant-scoped
  registry `delete`, and an `IncrementalIngestor.delete_document` orchestration.

### Out of scope (later / other sub-projects)
- **Collection *management* endpoints** (create/rename/list collections as first-class
  entities, move a document between collections). A collection here is a lightweight opaque
  label stamped at upload; there is no `collections` table. Full collection CRUD is a later
  increment if a real need appears.
- **Re-assigning `collection_id` after upload** (would require re-stamping every chunk's
  payload). Assignment is upload-time only.
- **Multi-collection membership** — a document belongs to at most one collection.
- **Bulk delete** (`DELETE` by collection) — single-document delete only for now.
- **D — Eval rework** and the **semantic cache** — unchanged from the A+B spec's deferrals.

## 3. Decisions (from brainstorming)

- **One combined spec** for document management *and* scoped retrieval — they share the
  `collection_id` thread through the stores, so splitting them would duplicate that work.
- **Collections now**, not per-`doc_id`-list filtering. The filter targets a `collection_id`
  the caller assigns at upload, rather than an ad-hoc list of document ids per query.
- **Asynchronous DELETE (`202`)** mirroring upload: a `deleting` status + an `arq` delete job,
  rather than an inline synchronous purge. Keeps the request fast and the store writes off the
  request path, consistent with ingest.

## 4. Design

### 4.1 Data model
- **`collection_id: str = ""`** added to `Document`, `Chunk` (`core/types.py`) and
  `DocumentRecord`. Empty string = *unassigned*. It is a first-class field (peer of
  `tenant_id`), **not** a `metadata` dict entry, so the stores can index/filter it. It is an
  opaque caller token: validated as a bounded string (≤ 128 chars, no control characters) and
  **never used in a filesystem path** — blob keys remain `sha256(tenant_id)/document_id`, so
  `collection_id` carries no path-traversal risk.
- **`DocumentStatus.DELETING = "deleting"`** added. Lifecycle:
  `processing → ready | failed`, and `ready | failed → deleting → (row removed)`. A delete
  that errors mid-purge lands the row in `failed` (with an error tag), never a half-deleted
  ghost.

### 4.2 Filters live in `retrieval/acl.py`
The ACL filter builders are the single home for "what a search is allowed / asked to see."
Extend both, keeping ACL mandatory and `collection_id` an optional additional `AND`:
- `qdrant_filter(acl, *, collection_id=None)` → append a `must` `FieldCondition(key="collection_id", …)` when set.
- `pg_where(acl, *, collection_id=None)` → append `AND collection_id = %s` when set.

`collection_id` can therefore only ever **narrow within a tenant**; it is structurally
impossible for it to widen or cross the tenant/ACL boundary.

### 4.3 Store changes (the cross-cutting part)
- **Qdrant** (`providers/vectorstores/qdrant_store.py`): add `collection_id` to the point
  payload (`_payload_from_chunk` / `_chunk_from_payload`) and a payload index for it; pass the
  extended filter through `search`.
- **pgvector** (`providers/vectorstores/pgvector_store.py`): add a `collection_id` column
  (DDL + INSERT + SELECT) and thread it into the WHERE via `pg_where`.
- **BM25 + in-memory** (`providers/sparse/*`, test fake): apply a `collection_id` equality
  check in the same Python pass that enforces ACL — after the ACL gate, before scoring.
- **`VectorStore.search` / `SparseRetriever.search`** signatures gain a keyword-only
  `collection_id: str | None = None`. Default `None` = no collection filter (identical to
  today's behaviour), so every existing caller is unaffected.

### 4.4 Registry & ingestor
- **Registry** (`providers/docstore/{memory,postgres}.py`): add
  `delete(document_id, tenant_id) -> None` — tenant-scoped hard delete, idempotent (deleting
  an absent row is a no-op). Postgres also gains the `collection_id` column (DDL + INSERT +
  SELECT + the `_row` mapping); the registry stores and returns `collection_id`.
- **`IncrementalIngestor.delete_document(tenant_id, doc_id, acl)`**: new orchestration that
  reuses the *already-tested* delete branch inside `ingest_document` — load the manifest for
  the doc, `store.delete(chunk_ids, acl)` + `sparse.delete(chunk_ids, acl)`, then
  `manifest.delete(tenant_id, doc_id)`. Idempotent: a doc with no manifest deletes cleanly.

### 4.5 Worker
- **`run_delete(deps, document_id)`** — pure body mirroring `run_ingest`, fail-closed. Order:
  `registry.get_privileged` → `ingestor.delete_document(tenant, doc_id, acl)` →
  `blobs.delete(rec.blob_key)` → **`registry.delete(document_id, tenant)` LAST**. Removing the
  registry row last means a crash mid-purge leaves the row visible (in `deleting`/`failed`)
  and the operation is safe to retry, since every underlying delete is idempotent. On
  exception → `set_status(FAILED, error=…)`.
- **`delete_document` arq task** added to `WorkerSettings.functions` alongside
  `ingest_document`.

### 4.6 Enqueuer (two job types)
Generalize the injected enqueuer to `enqueue(document_id, action="ingest")`, mapping `action`
to the arq function name (`"ingest" → ingest_document`, `"delete" → delete_document`). Upload
keeps calling it with the default; delete passes `action="delete"`. The default keeps existing
upload call sites and test fakes working.

### 4.7 API surface (`app/documents.py`, `app/api.py`)
- **`POST /documents`** — gains an optional `collection_id` form field (default `""`),
  validated (§4.1), stored on the `DocumentRecord`, and flowed to the worker → chunks.
- **`GET /documents`** — tenant-scoped list of summaries
  (`document_id, filename, content_type, size_bytes, status, chunk_count, collection_id,
  error`) via `registry.list(tenant)`; optional `?collection_id=` narrows it.
- **`DELETE /documents/{id}`** — `registry.get(id, tenant)` (→ `404` if absent/cross-tenant)
  → `set_status(DELETING)` → `enqueue(id, action="delete")` → **`202`**. Idempotent.
- **`POST /query`** (`app/api.py`) — `QueryRequest` gains optional `collection_id`. Threaded:
  `pipeline.run(question, acl, collection_id=…)` → `RAGPipeline.answer` → `Query.collection_id`
  → `HybridRetriever` → both `vector_store.search` and `sparse.search`.

### 4.8 Query object & retriever
- **`Query`** (`core/types.py`) gains `collection_id: str | None = None`.
- **`HybridRetriever.retrieve`** passes `query.collection_id` into both `search` calls. No
  fusion/rerank logic changes — the filter is applied *pre-similarity* in each store, so
  `top_k` is computed over already-scoped candidates (never a post-`top_k` trim that would
  silently drop hits).

## 5. Data flow

**Upload with collection**
`POST /documents {file, collection_id}` → record `{…, collection_id, status: processing}` →
enqueue ingest → worker stamps `collection_id` on parsed `Document`s → `chunk_document` copies
it to each `Chunk` → ingestor upserts chunks carrying `collection_id` into the vector payload +
BM25 → `status: ready`.

**Scoped query**
`POST /query {question, collection_id}` → `Query(text, acl, collection_id)` → `HybridRetriever`
→ dense + sparse both filter `tenant AND acl AND collection_id` pre-similarity → RRF → rerank →
answer grounded only in that collection.

**Delete**
`DELETE /documents/{id}` → `deleting` + enqueue delete → `202` → worker
`delete_document` (chunks dense+sparse) → `manifest.delete` → `blobs.delete` →
`registry.delete` → document gone from list and retrieval.

## 6. Error handling & security
- **Tenant isolation is fail-closed and structural.** `collection_id` is *always* ANDed under
  the tenant + ACL predicate in the same filter builder; it can only narrow within a tenant.
  Registry `delete` / `list` / `get` stay tenant-scoped (the worker's `get_privileged` is the
  only tenant-unscoped read, and it acts on the tenant the row was created under).
- **DELETE**: `404` for absent or cross-tenant id; `202` on accept; re-deleting a `deleting`
  doc re-enqueues (idempotent). A cross-tenant delete is a `404`, never a silent no-op that
  leaks existence.
- **Upload / query**: oversized or malformed `collection_id` → `422`; upload `415` / `413`
  unchanged. Empty `collection_id` on a query = no collection filter.
- **Push-down filtering** only — filtering after `top_k` would silently under-return; the
  design forbids it.

## 7. Testing
- **Stores**: each store (`qdrant` live-gated, `pgvector` live-gated, `bm25`, in-memory fake)
  filters by `collection_id`; a query scoped to collection A never returns collection B; a
  query with no collection returns both. Cross-tenant isolation retained with a collection set.
- **Registry**: `delete` removes only the tenant's row and is idempotent; `collection_id`
  round-trips through create/get/list; postgres DDL adds the column.
- **Ingestor**: `delete_document` removes exactly the doc's chunks from dense + sparse +
  manifest, leaving other docs intact; deleting a doc with no manifest is a clean no-op.
- **Worker**: `run_delete` happy path (chunks/blob/row gone), fail-closed (error → `failed`,
  no half state), and idempotent re-run.
- **Retriever**: `HybridRetriever` honours `Query.collection_id` on both dense and sparse legs.
- **API**: `GET /documents` (list + `?collection_id=` filter, tenant-scoped); `DELETE` (`202`,
  status → `deleting`, enqueues delete action, `404` cross-tenant); upload-with-collection;
  query-with-collection.
- **End-to-end**: upload two docs into collections A and B → query scoped to A returns only
  A's doc → `DELETE` A's doc → run the delete worker → A's doc is gone from both `GET /documents`
  and retrieval, B untouched.

## 8. Interfaces changed (summary)
- `core/types.py`: `+collection_id` on `Document`, `Chunk`, `DocumentRecord`, `Query`;
  `+DocumentStatus.DELETING`.
- `retrieval/acl.py`: `qdrant_filter` / `pg_where` gain `collection_id`.
- `providers/vectorstores/{qdrant,pgvector}_store.py`, `providers/sparse/*`, test fake:
  `search(..., collection_id=None)` + payload/column/DDL.
- `providers/docstore/{memory,postgres}.py`: `+delete`, `+collection_id`.
- `ingest/incremental.py`: `+delete_document`.
- `ingest/chunking.py`: copy `collection_id` doc → chunk.
- `ingest/worker.py`: `+run_delete`, `+delete_document` task, `WorkerSettings.functions`.
- `retrieval/hybrid.py`: thread `collection_id` into both `search` calls.
- `core/pipeline.py`: `run` / `answer` accept `collection_id`, build it into `Query`.
- `app/documents.py`: `GET /documents`, `DELETE /documents/{id}`, upload `collection_id`,
  generalized enqueuer.
- `app/api.py`: `QueryRequest.collection_id`, thread to `pipeline.run`.
