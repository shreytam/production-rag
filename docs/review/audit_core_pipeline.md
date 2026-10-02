# Core Pipeline Audit — Production RAG

**Scope:** `core/*`, `ingest/*` (+ parsers), `retrieval/*`, and the core-facing providers (`embedders/openai_compatible.py`, `vectorstores/qdrant_store.py`, `sparse/bm25.py` + `tenant_store.py` + `pickle_loader.py`, `rerankers/*`). Adjacent files (`cache/semantic_cache.py`, `ingest/audit.py`) were read where the query path depends on them.
**Method:** Static review at branch `ui-test-console` (HEAD `3af99ff`), cross-checked with grep over first-party code only. Read-only; nothing executed, installed, or modified.
**Reviewer:** ox-alpha subagent, 2026-08-24.

## Verdict

This is an unusually well-layered RAG codebase whose *security architecture* is ahead of typical practice: ACL is enforced pre-similarity inside every store, tenancy is structurally isolated in the sparse leg, IDs are deterministic enough to make ingestion idempotent, and the PII/compliance posture (value-free audit records, salt enforcement, fail-closed ingest) is deliberate. The defects cluster in **consistency between layers**, not in any single module: the semantic cache ignores ACL tags, incremental re-ingest does not propagate ACL changes to unchanged chunks, the reranker has no degraded-mode fallback, two divergent sparse-index persistence systems coexist (one of which the production retriever never reads), and the per-process BM25 cache breaks read-your-writes across the API/worker process boundary. Several advertised config knobs are dead. All findings below carry file:line evidence.

## Contents

- [Strengths](#strengths)
- [Findings](#findings)
  - [🔴 Critical](#critical)
  - [🟠 High](#high)
  - [🟡 Medium](#medium)
  - [⚪ Minor](#minor)
- [Sequenced remediation](#sequenced-remediation)
- [Verification boundary](#verification-boundary)

## Strengths

1. **Clean contract layering.** `core/interfaces.py` defines `@runtime_checkable` Protocols for every component; `core/registry.py:5-16` is the only module naming concrete classes, with lazy imports so a missing provider fails only when requested. Swapping providers is config-only by design and the code actually keeps that promise.
2. **ACL enforcement is structurally sound on the read path.**
   - Dense: `retrieval/acl.py:20-67` builds a Qdrant filter applied as `query_filter` *before* similarity (`providers/vectorstores/qdrant_store.py:133-139`) — no post-ANN filtering. The `acl_open` boolean + tag-overlap `should` branch faithfully models `ACLContext.allows()` (`core/types.py:37-47`), including the "no-tag caller sees only open chunks" edge.
   - Sparse: per-tenant index isolation is structural — a query can only ever touch its own tenant's BM25 index (`providers/sparse/bm25.py:53-69`); tag scoping runs filter→sort→top-k in the correct order (`bm25.py:72-82`), avoiding the classic filter-after-limit bug.
   - Mutations are ACL-scoped too: `delete`/`update_metadata` combine `HasIdCondition` with the ACL filter (`qdrant_store.py:153-173`).
   - Tenant-controlled filenames are SHA-256 hashed to defeat path traversal (`providers/sparse/tenant_store.py:16-29`, documented).
3. **Idempotency and crash recovery are designed in.** Deterministic chunk IDs `{doc_id}::{ordinal:06d}` (`ingest/chunking.py:118`) + deterministic UUID5 point mapping (`qdrant_store.py:24-25`) make upserts replayable; the incremental ingestor diffs by content hash and saves the manifest **last** so a crash re-applies the same delta (`ingest/incremental.py:40-71`). Worker delete order is stores → blob → registry-row-last (`ingest/worker.py:102-120`).
4. **Cache invalidation is wired into the write path, not left to TTL.** `_invalidate_caches` evicts both cache tiers across both the document's collection partition *and* the unscoped partition, with a collapse-to-one-scope guard and never-fail semantics (`ingest/worker.py:35-60`). This is more careful than most production systems manage.
5. **Contextual retrieval follows the Anthropic pattern with real defenses.** Delimiter spotlighting against prompt injection (`ingest/contextual.py:26-48`), PII post-scan of LLM output in redact mode (`contextual.py:113-118`), disk cache namespaced by `pii_mode` so redact/keep runs can't cross-contaminate (`contextual.py:77-79`). Both dense and sparse legs consistently index `embed_text` (prefix included) — `bm25.py:50` / `chunk.embed_text` — so hybrid fusion compares like with like.
6. **PII/compliance posture is deliberate.** Audit records are value-free by default (type/start/end only) with an optional salted hash gated by a config validator that refuses hashing without salt (`ingest/audit.py:42-48`, `core/config.py:245-251`); ingest fails closed if detection/audit errors (`ingest/run.py:105-109,135-137`); `redact()` merges overlapping spans instead of dropping tails (`ingest/pii.py:18-27`).
7. **Fail-fast production config.** `app_env=prod` requires JWT secret/JWKS/issuer/audience and forbids the dev signer (`core/config.py:231-243`). The OpenAI-compatible router/fallback validators remove a whole class of "one role forgot its key" misconfigurations.
8. **Guardrail ordering protects telemetry.** Input guards run *before* the root Langfuse span so unredacted questions never enter span parameters (`core/pipeline.py:112-140`); output-block scrubbing strips text, citations, contexts, and per-result reasons from the returned object (`pipeline.py:274-286`).
9. **RRF is correct and deterministic.** Proper `1/(k+rank)` fusion with explicit tie-breaking on `(−score, chunk_id)` both within lists and in the final ordering, plus per-source `component_scores` provenance (`core/rrf.py:25-49`).
10. **Test depth in this area.** Dedicated offline suites for RRF ties (`tests/test_sp4_rrf_ties.py`), chunker token conservation (`test_sp4_chunker_conservation.py`), tokenizer budget (`test_sp4_tokenizer_budget.py`), multitenant isolation (`test_multitenant_isolation.py`), store-mutation ACL (`test_store_mutations.py`), and tenant-store persistence (`test_tenant_sparse_store.py`).
11. **Honest self-documentation.** Removed gates are labeled as such rather than silently dead (`core/config.py:96-100` admits `hybrid_require_sparse` is now inert), and test seams (reranker scorer injection, generator injection into `build_query_rewriter`) keep the suite offline.

## Findings

Severity reflects worst-case impact assuming the code paths are used as the type system invites (e.g. documents with `acl_tags`, cache enabled). Several high-severity items are *latent* — masked today because every current ingestion path happens to write empty `acl_tags` (`ingest/worker.py:77`, `ingest/base.py:55`).

### 🔴 Critical

**C1. Semantic cache ignores the caller's ACL tags → within-tenant cross-user disclosure of tag-restricted content.**
- Evidence: `cache/semantic_cache.py:27-35` — `lookup/store` key on `(tenant_id, collection_id, embedding)` only. `core/pipeline.py:160-169` looks up the answer tier and returns a cached `Answer` whose `contexts`/`citations` were produced for *another user's* authorization scope; `pipeline.py:181-196` does the same for the retrieval tier, feeding other users' `ScoredChunk`s straight into generation.
- Impact: user A (tag `hr`) asks Q; the answer citing HR-only chunks is cached under `(tenant, collection, vec)`. User B (no tags, same tenant) with a semantically similar question receives A's cached answer **without ever passing an ACL check against those chunks**. The store-level ACL work (strength #2) is bypassed entirely above the stores. Latent only because current ingests set no tags; JWT claims already carry tags (`max_acl_tags`, `core/config.py:128`), so enabling the feature the types advertise arms this.
- Fix: include a digest of the caller's sorted `acl_tags` (or effective visibility class) in the cache identity of both tiers; alternatively partition the cache per acl-tag-set. Effort: M.

### 🟠 High

**H1. Incremental re-ingest never propagates ACL/tag changes to unchanged chunks — tightened access controls silently don't take effect.**
- Evidence: `_meta_hash` deliberately includes `title:tenant_id:acl_tags` (`ingest/incremental.py:14-16`), so the diff *detects* a tag change — but the metadata path writes only `{"title": c.title}` (`incremental.py:50-51`). `QdrantVectorStore.update_metadata` patches whatever dict it is given (`qdrant_store.py:164-173`); `acl_tags`/`acl_open` payload fields stay stale.
- Impact: re-uploading a doc with narrowed tags leaves the old, broad `acl_open=true` / old tag list on every text-unchanged point → documents remain visible to callers who should have lost access. Security drift with a green "re-ingest succeeded" signal. Related edge: changing `tenant_id` for the same `doc_id` orphans all old-tenant points forever (manifest lookup is per-tenant, `incremental.py:33`; nothing deletes across tenants).
- Fix: on meta-hash mismatch, upsert full ACL payload (`acl_tags`, `acl_open`, `collection_id`), not just title; treat tenant changes as delete+create. Effort: S.

**H2. Reranker outage degrades to wrong answers instead of degraded ranking — and a malformed response zeroes retrieval.**
- Evidence: `providers/rerankers/nim_rerank.py:62` `raise_for_status()` propagates (only timeouts/network errors retry, lines 31-36); `HybridRetriever.retrieve` has no try/except (`retrieval/hybrid.py:57-61`). If NIM returns 200 with missing/empty `rankings`, `normalize_candidates` receives `[]` and returns `[]` (`providers/rerankers/_common.py:16-29`) → fused candidates discarded → `retrieve()` returns `[]` → generation answers from zero context with no signal (`core/pipeline.py:204-213`).
- Impact: a NIM ranking-endpoint blip either 500s every query or — worse — produces confident "I don't know" answers fleet-wide while dense+BM25 results were fine. Best practice is fail-soft: log, fall back to RRF order (the pre-rerank window), mark `degraded=true` in response metadata.
- Fix: wrap `reranker.rerank` in HybridRetriever; on exception *or* implausibly empty output vs non-empty input, return `window[:rerank_top_n]` with source FUSED and set a metadata flag. Effort: S.

**H3. Cross-process read-your-writes is broken for the sparse leg: the API caches tenant BM25 indexes forever; the arq worker's writes are never re-read.**
- Evidence: `TenantSparseStore._retriever` memoizes per process and reloads only when the tenant is absent from `_cache` (`providers/sparse/tenant_store.py:31-40`). The API builds its pipeline once at startup (`app/api.py:27-33`), so its `TenantSparseStore` lives for process lifetime; the worker is a separate process persisting via atomic file replace (`tenant_store.py:42-47`). The module docstring claims "a separate worker process and the API process share one on-disk source" (`tenant_store.py:13-14`) — the disk is shared, the memory is not.
- Impact: after any incremental ingest/update/delete, previously-queried tenants keep retrieving deleted/stale chunks via BM25 until API restart (new docs simply invisible to the sparse leg; deleted docs still retrievable — the latter also undermines deletion promises). Dense leg stays fresh, so hybrid fusion mixes epochs.
- Also: `_save` uses a fixed `<hash>.tmp` name (`tenant_store.py:45`) — two processes saving the same tenant can interleave truncate/write/replace, potentially publishing a partial pickle. Use unique temp names (pid/uuid) or a lock.
- Fix: stat/mtime-check before using a cached tenant index, or pub/sub invalidation, or TTL on `_cache`; unique tmp suffix. Effort: M.

**H4. CLI batch ingest populates a legacy sparse-pickle that the production query path never reads — corpora ingested via `python -m ingest.run` run hybrid retrieval with a silently empty BM25 leg.**
- Evidence: `ingest/run.py:167-175` builds `BM25Retriever()` and pickles it to `.cache/bm25_{dataset}_{store}.pkl`. The production retriever resolves sparse exclusively through `TenantSparseStore` (`core/pipeline.py:347-351`, `core/registry.py:60-67`); `PickleSparseIndexLoader` is reachable only via `build_sparse_retriever(corpus=...)`, whose sole caller is a test (`tests/test_sp4_loader.py:1-13`).
- Impact: two divergent persistence systems for the same data. Anything ingested by the CLI has no tenant sparse index at query time; `BM25Retriever.search` returns `[]` for unknown tenants (`bm25.py:62-63`) and RRF quietly degenerates to dense-only — no error, no metric. Retrieval-quality regressions would be invisible.
- Fix: make `run.py` route through `IncrementalIngestor`/`TenantSparseStore` like the worker, and delete or fence the pickle path. Effort: S/M.

### 🟡 Medium

**M1. Pickle deserialization of index files is unauthenticated, unvalidated code execution surface.**
- `tenant_store.py:37` `pickle.loads(path.read_bytes())`; `run.py:172-174` writes whole-retriever pickles; `pickle_loader.py:24-26` loads `.cache/bm25_*.pkl`. Its isinstance checks run *after* `pickle.load`, i.e., after arbitrary code has already executed — validation theater. Any write primitive or tampered volume ⇒ RCE in the API/worker process. No HMAC/signature. Effort to fix: S (switch snapshot format to JSONL of Chunk models, or add an HMAC keyed server-side).

**M2. Embedding dimension and truncation are trusted, never verified.**
- `embed_dimension` is pure config (`openai_compatible.py:35-37`); response vector lengths are never checked against it or against the existing collection schema — `ensure_collection` creates only if absent and never validates an existing collection's vector size (`qdrant_store.py:74-84`). A model swap surfaces as cryptic Qdrant rejections or, with matching dim but different model space, silent retrieval garbage. Separately, NIM `truncate=END` (`openai_compatible.py:48-50`) silently drops long-chunk tails — quality loss with zero telemetry. Log/measure truncation when the API exposes usage/truncation info, and assert `len(vec) == dimension`. Effort: S.

**M3. Advertised config knobs are dead — operators tuning them change nothing.**
- `max_chunks_per_corpus` (`config.py:146`, "respect NIM rate limits"): zero usages outside its definition — corpora are unbounded.
- `chunk_overlap` (`config.py:105`): both call sites invoke `chunk_document(doc)` with defaults, hardcoding max_tokens=256/overlap=32 (`ingest/run.py:116`, `ingest/worker.py:84`; defaults at `chunking.py:66-68`). Config says 200.
- `hybrid_require_sparse` (`config.py:96-100`): self-documented no-op retained "for backward compatibility" — a trap for anyone reasoning about fail-closed behavior. Effort: S (wire knobs through or delete them).

**M4. Worker tasks swallow failures after marking FAILED — no retry policy for transient provider outages.**
- `run_ingest`/`run_delete` catch everything, record `FAILED` with only `type(e).__name__` (`ingest/worker.py:96-99,117-120`), and return normally, so arq sees success: no requeue, backoff, or dead-letter. One NIM 429 storm permanently fails every in-flight document until a human re-triggers. Also `error=type(e).__name__` discards message/traceback from the record. Effort: S (let arq retry transient classes, or raise after marking failed with retryable classification).

**M5. Store mutations use caller-*visibility* semantics where ownership semantics are needed — latent inability to manage tagged content.**
- The worker acts as owner with `ACLContext(tenant, ())` (`worker.py:72,111`); `qdrant_filter` for a no-tag caller matches only `acl_open==true` points (`retrieval/acl.py:39-50`), so once any ingestion path writes tagged chunks, worker `delete`/`update_metadata` silently no-op on exactly those points (FilterSelector matches nothing). Today it's masked (all tags empty); the moment C1/H1's tag machinery gets real use, deletions start ghosting. Sparse side differs again: `BM25Retriever.delete` ignores tags entirely (`bm25.py:113-119`). Three mutation scopes (visibility/owner/none) should be one documented rule. Effort: M.

**M6. Contextual prefixer: unbounded prompt growth, fragile cache reads, detector rebuilt per chunk.**
- Full `doc_text` ships in every per-chunk request with no length cap (`contextual.py:101-110`) — cost scales O(chunks × doc_length) and oversized docs blow the context window mid-ingest. `json.loads(cache_file.read_text())` has no guard (`contextual.py:97-99`): one crash-truncated cache file (writes aren't atomic, line 120) raises into ingest and fails the document. `build_pii_detector(...)` is constructed inside `prefix_for` per chunk in redact mode (`contextual.py:114-118`) — Presidio init per chunk. Cache key omits prompt/model version (`contextual.py:51-53`) although `DocManifest.prompt_version` exists precisely for that. Effort: S each.

**M7. Qdrant client: no API-key support, no timeout, broad excepts on index creation.**
- `QdrantClient(url=settings.qdrant_url)` only (`qdrant_store.py:71`) — no `api_key` param exists anywhere in Settings, ruling out Qdrant Cloud without code change; no client timeout configured; `ensure_collection` swallows *all* exceptions on `create_payload_index` (`qdrant_store.py:87-106`), including auth/consistency errors, mislabeled "already exists". `upsert` doesn't pass `wait=True`, so read-after-write consistency depends on client defaults worth pinning explicitly. Also `count(acl=None)` crosses tenants — safe today (internal use) but easy to misuse. Effort: S.

### ⚪ Minor

| # | Finding | Evidence | Why it matters |
|---|---------|----------|----------------|
| m1 | Naive BM25 tokenizer (whitespace+lowercase, no punctuation/stopword handling); `get_scores` scans the whole tenant corpus O(N) per query | `bm25.py:22-24,70` | Recall tax on punctuation-adjacent queries; linear scaling caps corpus size per tenant |
| m2 | `fuse_window=40` hardcoded; NIM `_TIMEOUT=10.0` class constant — neither reachable from Settings | `retrieval/hybrid.py:40`; `nim_rerank.py:24` | Config surface promises "every swappable knob" (`config.py:1`) |
| m3 | Fresh `httpx.Client` per rerank call | `nim_rerank.py:56` | Connection churn/TLS handshake per query |
| m4 | Reranker sends raw `chunk.text`, dropping the contextual prefix that dense/sparse indexed; reranked results lose `component_scores` provenance | `nim_rerank.py:49`; `_common.py:21-28` | Mild leg inconsistency; fusion provenance untraceable post-rerank for debugging |
| m5 | Tokenizer guessing maps all non-OpenAI models (e.g. Llama on NIM) to `cl100k_base`, and default safety margin is 0.0 | `context_assembly.py:27-35`; `config.py:104` | Token budget is approximate exactly where the deployed models aren't GPT-family |
| m6 | Final leftover chunk silently dropped when ≤ overlap tokens | `chunking.py:111-113` | Up to ~32 tokens of document tail never indexed |
| m7 | Local cross-encoder lazy-loads (and possibly HF-downloads) on first production query | `local_cross_encoder.py:32-39` | Multi-second/minute cold start on first request after deploy |
| m8 | Parser registry exact-matches content type (`text/plain; charset=utf-8` fails unless normalized upstream); `unstructured.partition.auto` has no timeout/page cap | `parsers/base.py:27-34`; `unstructured_parser.py:17` | A 25 MiB scanned PDF can pin a worker for minutes with no bound |
| m9 | Golden eval export writes doc IDs under key `relevant_chunk_ids` | `ingest/run.py:190-193` | Field name lies; metric code must remember to misread it |
| m10 | `pg_dsn` default embeds a placeholder password string | `config.py:46` | Cosmetic, but invites copy-pasting fake creds |
| m11 | Worst-case sync call path: 600 s timeout × 5 SDK retries per provider call, sequential dense+sparse legs | `config.py:87-88`; `hybrid.py:50-56` | A hung provider stalls one worker thread for many minutes; no overall query deadline |

## Sequenced remediation

Order matters here: some fixes change authorization behavior and must land together with their tests.

1. **C1 + H1 (same theme: tag-aware ACL end-to-end).** Fix cache identity to include caller tag-digest *and* fix incremental metadata propagation in the same change-set — otherwise fixing one exposes the other (fresh uncached queries would hit stale tagged payloads). Add a regression test: ingest tagged doc → tighten tags → re-ingest → query as both principals, cached and uncached.
2. **H2 (rerank fail-soft).** Small, isolated, immediately reduces outage blast radius. Add a fake flaky reranker test asserting fallback-to-fused order.
3. **H3 + M5 (sparse coherence + mutation semantics).** Decide one mutation-scope rule (recommend tenant-owner scope for worker mutations), then add mtime/invalidation for cross-process freshness. These interact: invalidation tests depend on deletes actually matching.
4. **H4 + M1 + M3 (retire legacy sparse/pickle path).** Route CLI ingest through the IncrementalIngestor, delete `PickleSparseIndexLoader` + corpus pickles, wire or remove `max_chunks_per_corpus` / `chunk_overlap` / `hybrid_require_sparse`. One PR removes an entire risk class.
5. **M2/M4/M6/M7** are independent small hardening tasks (dimension asserts, arq retry policy, prefixer guards, Qdrant client options).

## Verification boundary

Static review only. I did **not**: execute the pipeline or pytest, connect to Qdrant/Redis/NIM, run ingestion, or observe runtime behavior of the RedisVL cache backend — findings about caching/staleness are derived from code paths and process-topology reasoning (API builds once at `app/api.py:27-33`; worker is a separate arq process), not from a live reproduction. Line numbers reference branch `ui-test-console` @ `3af99ff`. The untracked `docs/PRODUCTION_READINESS_AUDIT.md` was not cross-checked (out of scope for this area audit).
