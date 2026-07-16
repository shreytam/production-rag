# Product Ingestion — API-driven, Async, Incremental — Design

_Date: 2026-07-17_

## 1. Motivation

Today the system is **eval-driven**: ingestion is a CLI script (`python -m ingest.run
--dataset hotpotqa`) hard-wired to three academic benchmark corpora (HotpotQA, arXiv,
FinanceBench). Those `DatasetAdapter` classes exist to produce gold Q/A pairs
(`build_golden()`) for the SP5 eval gate — they are *scorers*, not document *loaders*.

We are pivoting toward a **product RAG**: a caller uploads a real document (PDF and other
formats) to the API; the system parses → PII-handles → chunks → embeds → stores it scoped
to their tenant; once `ready`, their existing `/query` retrieves over it. Ingestion becomes
**API-driven and asynchronous**, not a CLI over static datasets.

This document specifies the **first shippable slice** of that pivot.

## 2. Scope

### In scope (this spec)
- A multi-format **document parser** subsystem behind a `DocumentParser` interface.
- An **async, API-driven ingest path**: upload endpoint, raw-file storage, a document
  registry with lifecycle status, and a Redis-backed `arq` worker.
- The **SP8 incremental engine** pieces this path depends on (`VectorStore.delete` /
  `update_metadata`, `DocManifest` / `ManifestStore`, `IncrementalIngestor`), **merged into
  this spec** rather than sequenced separately.
- **Per-tenant incremental BM25** — the sparse-index counterpart the SP8 plan does not
  cover.
- A **config change** so all model calls route through one OpenAI-compatible LLM router
  (base_url + api_key), with the reranker excluded.

### Out of scope (later sub-projects)
- **C — Scoped retrieval & document management:** optional per-document/collection filter
  on queries; `DELETE /documents/{id}` (purge vectors + BM25 + raw file); collections.
- **The semantic cache:** already fully planned in `docs/superpowers/plans/2026-06-27-rag-cache.md`
  (L1 exact / L2 semantic / L3 embedding / L4 retrieval on Redis, tenant-partitioned). It is a
  query-path feature and is built there, not here.
- **D — Eval rework:** replacing the HuggingFace academic adapters + auto-gold with
  hand-curated Langfuse Datasets driving the SP5 gate. Tracked separately; this spec does not
  remove the benchmark adapters.
- **SP12** query rewriting, **SP6** resilience, **SP7** cost/observability, **SP9**
  deployability (Docker/compose for the new worker) — pre-existing plans that layer on later.

## 3. Relationship to the existing roadmap

The plans in `docs/superpowers/plans/` are **future work** (git history stops at SP5). Two
overlap this spec and are reconciled here:

- **SP8 · Ingest Robustness & Incremental** already designs the incremental *engine* —
  tenant-scoped `VectorStore.delete` / `update_metadata`, a blake2b content-hash
  `DocManifest` / `ManifestStore`, and an `IncrementalIngestor` that diffs a document into
  embed / update / delete sets (writing the manifest strictly *after* store writes succeed —
  "D-ORDER"). SP8 drives this from the CLI over benchmark corpora. **This spec reuses that
  engine** and adds the API + parser + async worker on top, rather than reinventing it. Per
  the approved decision, the required SP8 pieces are **merged into this spec** so it is one
  self-contained plan. **Gap:** SP8's `IncrementalIngestor` only touches the dense store +
  manifest; it does not update BM25. This spec closes that gap (§4.6).
- **`rag-cache`** and **SP9's 64 KiB request-body → HTTP 413 rule** are noted as constraints
  (the upload endpoint needs its own larger size limit, exempt from the global rule).

## 4. Components

Every implementation sits behind an interface so it can be swapped and unit-tested in
isolation, consistent with the project rule that `core/registry.py` is the only module that
names concrete classes.

### 4.1 Config — single OpenAI-compatible router
Introduce one router base_url + api_key (e.g. `llm_base_url`, `llm_api_key`). The embeddings,
generation, contextual, and judge roles **inherit** it unless a role-specific override is
set. The **reranker is excluded** — NIM's rerank endpoint is not part of the OpenAI standard,
so it stays `local` (or keeps its own base_url). Result: pointing the whole stack at a router
(LiteLLM / OpenRouter / vLLM / NIM) is "set base_url + key once."

### 4.2 `DocumentParser` subsystem (sub-project A)
- Protocol: `parse(raw: bytes, filename: str, content_type: str) -> list[Document]`.
- One `UnstructuredParser` implementation covering the rich multi-format set (PDF, Office,
  HTML; OCR optional), with heavy imports kept lazy.
- A content-type / extension → parser registry, gated by an `ingest_allowed_types` allowlist.
- `max_upload_bytes` size limit enforced before work is scheduled.
- Chunk `metadata` carries parser-derived page / section info.

### 4.3 `BlobStore`
- Interface: `put(key, data) / get(key) -> bytes / delete(key)`.
- `LocalDiskBlobStore` (configurable dir) now; MinIO/S3 later with no API change.
- Stores the raw uploaded bytes so ingestion can be retried without a re-upload.

### 4.4 Document registry (Postgres `documents` table)
The durable, API-facing source of truth for lifecycle. Columns: `document_id (uuid PK)`,
`tenant_id`, `filename`, `content_type`, `size_bytes`, `status` (`processing|ready|failed`),
`error`, `chunk_count`, `blob_key`, `created_at`, `updated_at`. Uses the pgvector Postgres
already in `infra/docker-compose.yml` as a plain relational table. Complements SP8's manifest
(internal chunk-hash state) — the registry is the *external* status surface.

### 4.5 SP8 engine (merged)
- `VectorStore.delete(chunk_ids, acl)` and `update_metadata(updates, acl)` on the protocol
  and Qdrant/pgvector backends — tenant-scoped, fail-closed on cross-tenant.
- `DocManifest` / `ChunkRecord` types, `ManifestStore` protocol, `JsonlManifestStore`
  (atomic writes via temp + `os.replace`).
- `IncrementalIngestor.ingest_document(tenant_id, doc_id, chunks, acl)` — blake2b diff into
  embed / metadata-update / delete sets, manifest saved only after store writes succeed.

### 4.6 Per-tenant incremental BM25 (the SP8 gap)
Today the BM25 sparse index is a single pickle per corpus, rebuilt wholesale by
`sparse.index(all_chunks)` — incompatible with incremental per-tenant upload. This spec
introduces a **tenant-keyed sparse index** with incremental `add` / `delete` and persistence.
`IncrementalIngestor` updates it alongside the dense store. `HybridRetriever` selects the
tenant's index at **query time** from `query.acl.tenant_id`, replacing the current
build-time corpus binding in `core/pipeline.py`. This preserves the SP4 hybrid-retrieval
quality while respecting tenancy.

### 4.7 `arq` worker + Redis
A separate worker process runs `ingest_document(document_id)`. Redis (already in
`infra/docker-compose.yml`, currently serving Langfuse) is the broker. `arq` is async-native
(matches FastAPI) and adds no new service. Heavy parsing runs off the API process.

### 4.8 Ingest API routes
On the existing FastAPI app, all authed via `require_principal`:
- `POST /documents` (multipart) — validate type/size → `BlobStore.put` → registry
  `processing` → enqueue arq job → `202 {document_id, status}`. Has its **own** size limit,
  exempt from SP9's 64 KiB global body rule.
- `GET /documents/{id}` — status + metadata, **tenant-scoped** (404 if not the caller's
  tenant, so existence never leaks across tenants).
- `GET /documents` — list the caller's documents.

## 5. Data flow

### Upload
```
client ── POST /documents (JWT, multipart) ──▶ API
  API: validate content_type ∈ allowlist, size ≤ max_upload_bytes
       BlobStore.put(blob_key, raw)
       registry.insert(document_id, tenant_id=principal.tenant, status="processing")
       arq.enqueue("ingest_document", document_id)
  API ──▶ 202 { document_id, status: "processing" }
```

### Worker (`ingest_document`)
```
load registry row (tenant_id, blob_key, content_type, filename)
raw   = BlobStore.get(blob_key)
docs  = parser.parse(raw, filename, content_type)     # tenant attached from registry
docs  = apply PII policy (reuse ingest._apply_pii_ingest_policy)
chunks = chunk_document(doc) for doc in docs           # doc_id = document_id
chunks = optional contextual prefixing
IncrementalIngestor.ingest_document(tenant_id, doc_id, chunks, acl):
    embed delta → Qdrant.upsert → per-tenant BM25 add/delete → manifest.save (last)
registry.update(status="ready", chunk_count=N)
# on any exception: registry.update(status="failed", error=<sanitized>)
```

### Query (unchanged for callers)
`/query` runs the existing hybrid pipeline; internally the sparse retriever now resolves the
**tenant's** BM25 index at query time. A document is retrievable as soon as its status is
`ready`.

## 6. Error handling & security

- **Validation:** unsupported type → **415**; oversized → **413** (upload's own limit);
  empty body → **422**.
- **Enqueue failure** (Redis down) → **503**, fail-closed; the upload is never silently
  dropped or marked ready.
- **Worker failures** (parse / embed / store) → status `failed` + sanitized error string;
  no partial "ready". Manifest-after-writes (D-ORDER) plus arq retries are safe because
  `IncrementalIngestor` is **idempotent** (keyed by `doc_id` + content-hash) — a re-run
  converges instead of duplicating.
- **Tenant isolation:** chunk `tenant_id` / `acl_tags` derive from the verified principal
  captured at upload time and stored in the registry — **never** from file content.
  `delete` / `update_metadata` fail-closed on cross-tenant. `GET /documents/{id}` returns 404
  (not 403) for other tenants' ids.
- **PII:** the existing ingest PII policy runs in the worker before chunks are embedded or
  stored; keep mode tags chunks and audits, redact mode cleans doc text first.

## 7. Testing strategy

All tests run **offline** with fakes — no network, no live models.

- **Parser:** per-format fixture files → expected `Document` text/metadata; allowlist
  rejection; oversize rejection.
- **BlobStore:** put/get/delete round-trip; missing-key behavior.
- **Registry:** CRUD; tenant-scoped read returns 404 across tenants; status transitions.
- **SP8 engine:** `IncrementalIngestor` diff (embed/update/delete sets); manifest atomic
  write + reload; tenant-scoped store delete/update fail-closed.
- **Per-tenant BM25:** incremental add/delete; two tenants' indexes never bleed; query-time
  selection by `acl.tenant_id`.
- **Worker:** `ingest_document` with fake embedder / store / parser drives
  `processing → ready`; forced failure drives `processing → failed` with no partial index.
- **API:** upload → poll status; type/size rejection codes; cross-tenant 404; enqueue-failure
  503.
- **End-to-end:** upload a small fixture doc → `ready` → `/query` retrieves its chunk
  (tenant-scoped), using fakes.
- **Config:** all model roles inherit the single router base_url/key; a role override wins;
  the reranker does **not** inherit the router.

## 8. Open questions / follow-ups

- Whether the API-facing registry and SP8's internal manifest should later be unified into a
  single store (kept separate here: different lifetimes and consumers).
- Object storage: `LocalDiskBlobStore` now; swap to the existing MinIO for multi-node
  deployment (SP9).
- Deletion of a document's vectors + BM25 entries + raw blob is **sub-project C**, but
  `IncrementalIngestor` + `VectorStore.delete` + per-tenant BM25 `delete` already provide the
  primitives it will call.
