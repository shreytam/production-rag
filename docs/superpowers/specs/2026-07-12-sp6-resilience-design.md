# SP6 · Resilience & Failure Modes — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 1. First Phase-1 slice; depends on SP1 (verified `Principal`/`ACLContext` on `/query`) and SP2 (guardrails) being in place. Claims the `request_id` + catch-all exception handler that SP1 §8 and SP2 explicitly deferred here.

---

## 1. Context & problem

The query path issues **five external calls** (embed → dense store → sparse → RRF → rerank → generate) with essentially no failure isolation and no request-level timeout discipline. `/query` is defined as a **synchronous** handler (`def query`, `app/api.py:70`), so FastAPI/anyio runs it in a **bounded worker threadpool** (~40 threads by default). Any single call that blocks without a timeout holds a thread for the full hang; ~40 concurrent hangs exhaust the pool and take the whole service down — a slow backend becomes a total outage, not a slow query. This threadpool-saturation mechanism is the load-bearing reason timeouts matter more here than retries.

Verified defects (line numbers checked against current code):

| # | Defect | File:line | Impact |
|---|---|---|---|
| 1 | `HybridRetriever.retrieve` runs embed→dense→sparse→RRF→rerank with **no try/except and no degradation** — any single stage failure aborts the whole query | `retrieval/hybrid.py:48-56` | A reranker 429, an embed error, or a store hang fails the query with a 500 even when dense+sparse succeeded |
| 2 | `NIMReranker` retries **only** `TimeoutException`/`NetworkError` (`_is_transient`, and the `retry_if_exception_type` on the decorator). `response.raise_for_status()` throws `httpx.HTTPStatusError`, which is **not** caught — a 429/503 is **not retried** and propagates as a hard failure | `nim_rerank.py:11-12, 31-36, 62` | Transient upstream 429/5xx (the common overload signal) kills the query; `Retry-After` ignored |
| 3 | `QdrantClient` constructed with **no explicit `timeout`** (constructor default is `timeout=None`) | `qdrant_store.py:69` | A slow/hung Qdrant blocks the worker thread indefinitely → threadpool saturation |
| 4 | `psycopg.connect(dsn)` opens a **new connection with no `connect_timeout`, no `statement_timeout`, and no pool**, and runs `CREATE EXTENSION` **per query** | `pgvector_store.py:28-36, 50-51` | Per-query connection churn; a slow query hangs the thread for minutes; no server-side statement bound |
| 5 | No Qdrant/psycopg retry on transient errors | `qdrant_store.py:115`, `pgvector_store.py:147-150` | A single blip 5xx / dropped connection fails the query with a 500 |
| 6 | Generator + embedder use `openai.OpenAI(timeout=600.0, max_retries=5)` (`config.py:82-83`) — the SDK **already** retries 429/5xx with backoff and honors `Retry-After`, **but** a 600s ceiling × up to 5 retries can pin an interactive worker for many minutes | `generators/openai_compatible.py:31-36`, `embedders/openai_compatible.py:28-33`, `config.py:82-83` | Not a "missing retry" bug — a **timeout/budget** bug: the interactive path inherits an ingest-sized ceiling |
| 7 | **No global FastAPI exception handler**; any unhandled error returns FastAPI's default 500 with no `request_id`, no sanitization, no correlation for logs | `app/api.py` (absent) | Unhandled failures leak raw error text and are untraceable; SP1/SP2 deferred this here |

Note on #6: this spec does **not** re-add retries to the SDK-backed generator/embedder (they already have them). It tunes their **timeout and total-attempt budget** for the interactive path and standardizes retryable-error classification for the three raw-HTTP callers (NIM reranker, Qdrant, psycopg).

---

## 2. Goals

- **Bound every external call** with an explicit connect + operation timeout so no call can hold a worker thread indefinitely (kills the threadpool-saturation outage).
- **Bounded retries with backoff** on the three *raw* callers — NIM reranker, Qdrant, psycopg — retrying `429` and `5xx` and transient network/timeout errors, **honoring `Retry-After`**, with a capped total attempt budget.
- **Partial-failure degradation** in `HybridRetriever`: a failed stage degrades gracefully instead of aborting —
  - reranker fails → return the **RRF-fused order** (truncated to `rerank_top_n`);
  - sparse fails → **dense-only**;
  - embed fails → **sparse-only** (BM25 scores from `query.text`, not the vector, so it still runs);
  - both dense **and** sparse produce nothing → typed `UpstreamUnavailable` → fast **503**.
- **Retry exhaustion → fast 503** (`Retry-After` echoed when known), never a slow 500.
- **pgvector connection pooling** (`psycopg_pool`) with `register_vector` in the pool's per-connection configure hook and `CREATE EXTENSION` moved out of the hot path into `ensure_collection`.
- **Interactive timeout/budget tuning** for the SDK-backed generator + embedder (separate from the 600s ingest ceiling).
- **Global exception handler** returning sanitized JSON `{ "detail", "request_id" }` with a per-request `request_id` on every response (success and error), fulfilling the SP1/SP2 deferral.

## 3. Non-goals (deferred)

| Excluded concern | Owner |
|---|---|
| Rate limiting, request-size caps, CORS / TrustedHost, Dockerfile / deploy topology | **SP9 · Deployability** |
| Cost accounting / per-query budget enforcement | **SP7 · Cost & Quotas** |
| Auth / tenancy / `Principal` derivation | **SP1 · Security & Tenancy** (done) |
| Guardrail correctness & block-leak | **SP2 · Guardrail Correctness** (done) |
| Physical namespace-per-tenant isolation | VDB-Decision / SP11 |
| A full circuit breaker | **Optional here** (§11) — the always-on unit is timeouts + bounded retry + degradation; a breaker is a follow-on hook, kept out of core scope to keep the slice small |

Boundary note (to avoid contradicting SP1 §8): **SP6 owns** the global exception handler, the `request_id`, and the retry-exhaustion → 503 mapping. **SP9 owns** CORS/TrustedHost/rate-limiting/request-size. SP1 deferred exactly the `request_id` + catch-all sanitizer to "SP9/SP6"; this spec pins it to SP6.

---

## 4. Decisions (PROPOSED)

Each row leads with the best-practice option. **These are proposed for the user to confirm/override on review.**

| # | Decision | Choice (best practice) | Rationale |
|---|---|---|---|
| D1 | Where retry/backoff logic lives | Small `core/resilience.py` util (retryable predicate + `Retry-After`-aware wait + typed `UpstreamUnavailable`), mirroring the `core/rrf.py` precedent | Single source of retry policy; providers import it, no parallel framework |
| D2 | Retry library | **tenacity** (already a dependency) with a **custom `wait` callable** | `wait_exponential` alone ignores response headers; the custom wait reads `exc.response.headers["Retry-After"]` |
| D3 | Retryable classification | `429`, `5xx` (`HTTPStatusError` where `500 ≤ status ≤ 599` or `== 429`), plus `TimeoutException`/`NetworkError`; **never** 4xx except 429 | 4xx (auth/bad-request) are permanent — retrying wastes the worker |
| D4 | Degradation policy | Per-stage try/except in `HybridRetriever` with the fallbacks in §2; hard-fail **only** when dense+sparse both empty | Partial answer beats a 500 when a subset of stages succeeded |
| D5 | Timeout model | **Two-band timeouts**: interactive (`query_timeout_seconds`, small) for reranker + query-path stores + a distinct interactive generator budget; ingest keeps the existing 600s | An interactive user must not wait an ingest-sized ceiling |
| D6 | pgvector connectivity | **`psycopg_pool.ConnectionPool`** with a `configure` hook (`register_vector`), `connect_timeout`, and per-session `statement_timeout`; `CREATE EXTENSION` moved to `ensure_collection` | Removes per-query churn, bounds slow queries server-side, registers the vector adapter once per connection |
| D7 | Generator/embedder fix | **Do not add retries** (SDK already retries + honors `Retry-After`); add an **interactive timeout + total-attempt cap** distinct from ingest | Avoids duplicate/contradictory retry layers; fixes the real 600s-pin defect |
| D8 | Exhaustion → HTTP mapping | Typed `UpstreamUnavailable` → **503** with `Retry-After` header when known; unexpected errors → sanitized **500** with `request_id` | Fast, honest backpressure; no leaked internals |
| D9 | `request_id` source | Trust inbound `X-Request-Id` if present and well-formed (bounded length/charset), else generate a `uuid4`; echo it on every response | Correlates across an upstream gateway while staying injection-safe |
| D10 | Registry wiring | Only the **optional** circuit breaker (a swappable component) is registry-wired; timeouts/retries are provider-internal | Consistent with the registry contract — it names swappable components, not per-provider tuning |
| D11 | Circuit breaker | **Deferred to §11** (optional hook), off by default | Keeps the slice focused; degradation already prevents cascading failure for the common cases |

---

## 5. Architecture & components

Follows the existing pattern: Protocols in `core/interfaces.py`, concrete impls in `providers/`, concrete classes named only in `core/registry.py`, cross-cutting pure logic in a small `core/` util (like `core/rrf.py`). Small, single-purpose units.

### 5.1 `core/resilience.py` (new — pure util, no framework deps)
- `is_retryable(exc) -> bool` — `True` for `httpx.HTTPStatusError` with status `429` or `500–599`, and for `httpx.TimeoutException`/`httpx.NetworkError`. Everything else (incl. other 4xx) → `False`.
- `retry_after_wait(...)` — a tenacity `wait` **callable** that, on the last exception, reads `Retry-After` (seconds or HTTP-date) from `exc.response.headers` and returns that delay (capped at `retry_max_wait_seconds`); otherwise falls back to exponential backoff. Purely computed; no sleeping of its own.
- `resilient_retry(attempts, ...)` — a preconfigured `tenacity.retry` decorator factory bundling `is_retryable`, `retry_after_wait`, `stop_after_attempt`, and — on exhaustion — **re-raising** the underlying error so callers translate it.
- `UpstreamUnavailable(Exception)` — typed exhaustion / hard-degradation signal carrying an optional `retry_after: float | None` and a `stage: str`. The app layer maps it to 503.

`core/resilience.py` imports only `httpx` + `tenacity`; it does not import providers (no cycles).

### 5.2 `NIMReranker` — `providers/rerankers/nim_rerank.py` (modify)
- Replace the timeout/network-only `retry` decorator with `resilient_retry(attempts=settings-driven)` so **429/5xx are retried** with `Retry-After` honored.
- Keep `response.raise_for_status()` — its `HTTPStatusError` is now classified by `is_retryable`.
- Timeout comes from `query_timeout_seconds` (interactive band), replacing the hardcoded `_TIMEOUT = 10.0`.
- Constructor takes the resolved timeout + attempts from the registry (keeps the class config-free, consistent with the current constructor shape).

### 5.3 `QdrantVectorStore` — `providers/vectorstores/qdrant_store.py` (modify)
- Pass an explicit `timeout=settings.query_timeout_seconds` to `QdrantClient(...)` (constructor default is `None`).
- Wrap `search()` (and `count()`) network calls in `resilient_retry`. Qdrant raises `qdrant_client` exceptions wrapping `httpx` errors/`UnexpectedResponse`; the classifier is extended to recognize `qdrant_client.http.exceptions.UnexpectedResponse` (status ≥ 500 / 429) alongside the `httpx` types.
- Exhaustion in `search()` raises `UpstreamUnavailable(stage="dense")`, which `HybridRetriever` catches for degradation (dense failure alone is not fatal).

### 5.4 `PgVectorStore` — `providers/vectorstores/pgvector_store.py` (modify)
- Introduce a module-level (per-store-instance) **`psycopg_pool.ConnectionPool`** built from the DSN with `connect_timeout`, `max_size`, and a `configure=` hook that runs `register_vector(conn)` and `SET statement_timeout = <ms>` once per connection.
- `_conn()` becomes `pool.connection()` (context-managed checkout), replacing per-call `psycopg.connect`.
- **Move `CREATE EXTENSION vector` out of the checkout path** into `ensure_collection` (a one-time migration step); the pool's configure hook assumes the extension already exists.
- Wrap `search()`/`count()` in `resilient_retry`; classify `psycopg.OperationalError` (connection/timeouts) as retryable, `psycopg.errors.QueryCanceled` (statement_timeout hit) as an `UpstreamUnavailable(stage="dense")`. Programming errors are **not** retried.

### 5.5 `HybridRetriever` — `retrieval/hybrid.py` (modify) — degradation core
Per-stage isolation (single-purpose helper methods, still one `retrieve`):
```
embed:   try qvec = embed(text)            except → qvec = None (embed_failed)
dense:   if qvec: try dense = store.search except UpstreamUnavailable → dense = []
sparse:  try sparse = sparse.search        except UpstreamUnavailable → sparse = []
if not dense and not sparse:  raise UpstreamUnavailable(stage="retrieval")   # → 503
fused = RRF([dense, sparse])
rerank:  try return reranker.rerank(...)   except UpstreamUnavailable → return fused[:rerank_top_n]
```
Each caught degradation is recorded on the returned set / trace (`degraded_stages`) so the pipeline and observability can see it. `DenseRetriever` (baseline) is left unchanged except that its single store call now benefits from the store-level timeout/retry.

### 5.6 `app/errors.py` (new) + `app/api.py` (modify) — global handler & request_id
- `install_error_handlers(app)` registers:
  - a handler for `UpstreamUnavailable` → **503** `{ "detail": "Service temporarily unavailable", "request_id": ... }` + `Retry-After` header when known;
  - a catch-all `Exception` handler → **500** sanitized `{ "detail": "Internal error", "request_id": ... }` (no stack, no upstream text) and a `logger.exception` keyed by `request_id`.
- A lightweight middleware assigns `request.state.request_id` (inbound `X-Request-Id` if valid, else `uuid4`) and sets `X-Request-Id` on every response — so successes are correlatable too.
- Wired in `app/api.py` at app construction: `install_error_handlers(app)`.

### 5.7 Registry — `core/registry.py` (modify)
- `build_reranker` / `build_vector_store` pass the resolved interactive timeout + attempt budget into the concrete impls (still the only place classes are named).
- Circuit breaker (if elected later) is the only new **swappable** thing that would be registry-wired; not in this slice.

---

## 6. Data flow

```
POST /query  (X-Request-Id? )
  middleware: request_id = valid(X-Request-Id) or uuid4()   # set on request.state + response
  route → pipeline.run(question, acl)                        # acl from verified Principal (SP1)
    HybridRetriever.retrieve(query):
      embed(text) ──fail──▶ qvec=None
      dense = store.search(qvec, acl)   [timeout+retry(429/5xx,Retry-After)]  ──exhaust/skip──▶ []
      sparse = sparse.search(text, acl)                                       ──fail──▶ []
      if dense==[] and sparse==[]:  raise UpstreamUnavailable(stage="retrieval")  ─────────▶ 503
      fused = RRF([dense, sparse])
      rerank(fused[:window])            [timeout+retry(429/5xx,Retry-After)]  ──exhaust──▶ fused[:rerank_top_n]
    grounded.generate(...)              [SDK retry + interactive timeout/budget]
  ── UpstreamUnavailable ─▶ 503 { detail, request_id } (+ Retry-After)
  ── any other Exception ─▶ 500 { detail:"Internal error", request_id } (sanitized, logged)
  ── success ────────────▶ 200 { ..., degraded_stages? }  (X-Request-Id header always)
```

Degradation is **silent to correctness** (still ACL-scoped — fallbacks only ever *narrow* to already-tenant-scoped result subsets) and **visible to observability** (`degraded_stages` on the trace/response metadata).

---

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `query_timeout_seconds` | `15.0` | Interactive per-call timeout for reranker + query-path store calls (replaces the reranker's hardcoded 10s and Qdrant's `None`) |
| `retry_attempts` | `3` | Max total attempts (initial + retries) for the raw callers (reranker, Qdrant, pgvector) |
| `retry_min_wait_seconds` | `0.5` | Exponential backoff floor |
| `retry_max_wait_seconds` | `8.0` | Backoff ceiling **and** cap applied to an honored `Retry-After` |
| `gen_query_timeout_seconds` | `60.0` | Interactive generator timeout (distinct from the 600s ingest ceiling; used for the answer-synthesis call) |
| `gen_query_max_retries` | `2` | Interactive generator total-retry budget (SDK-level), separate from ingest `max_retries=5` |
| `pg_pool_min_size` | `1` | pgvector pool floor |
| `pg_pool_max_size` | `10` | pgvector pool ceiling (kept below the ~40 threadpool so DB is never the pool's own bottleneck) |
| `pg_connect_timeout_seconds` | `5.0` | psycopg `connect_timeout` |
| `pg_statement_timeout_ms` | `15000` | Per-session `SET statement_timeout` (server-side hard bound on slow queries) |
| `circuit_breaker_enabled` | `False` | Optional breaker (§11); off in this slice |
| `max_request_id_len` | `128` | Bound + charset-validate an inbound `X-Request-Id` before echoing it |

The existing `request_timeout_seconds` (600) and `max_retries` (5) are **retained for ingest** and no longer read by the interactive query path.

---

## 8. Error handling (fail closed on security; fail soft on availability)

- **Security invariant preserved under degradation.** Every fallback path (`dense=[]`, `sparse=[]`, RRF-order rerank fallback) returns a **subset** of an already ACL-scoped candidate set. No fallback ever widens scope, drops the ACL filter, or reads across tenants — degradation trades recall for availability, never isolation. A test asserts an org-A query with a failing reranker still returns **only** org-A chunks.
- **Classification is conservative.** Only `429`/`5xx`/transient-network/timeout are retried. 4xx (auth, malformed) and programming errors fail fast — retrying them wastes a worker and can amplify an outage.
- **Exhaustion is honest.** Retry exhaustion and the both-empty-retrieval case raise `UpstreamUnavailable` → **503** with `Retry-After` when known, never a masked 500 or an indefinite hang.
- **The catch-all sanitizes.** The global 500 handler emits a fixed `"Internal error"` string + `request_id` only — no exception text, no upstream response body, no stack. Full detail is logged server-side keyed by `request_id`.
- **No secret/PII leakage in logs.** Retry/exhaustion logs record stage, status code, attempt count, and `request_id` — never headers, tokens, DSNs, or query text.
- **`request_id` is injection-safe.** An inbound `X-Request-Id` is accepted only if it matches a bounded `[A-Za-z0-9._-]{1,max_request_id_len}` pattern; otherwise a fresh `uuid4` is used (prevents log-forging / header injection).
- **Pool exhaustion is bounded, not hung.** The pgvector pool checkout uses a bounded wait; a checkout timeout surfaces as `UpstreamUnavailable(stage="dense")` → degradation, not a stalled thread.

---

## 9. Testing (TDD) — concrete, offline-testable behaviors

All tests use fakes / `httpx.MockTransport` (or `respx`) — **no live network**.

**Retry / classification (`core/resilience.py`):**
- 429 then 200 → retried, second attempt returned.
- 503 then 200 → retried and succeeds.
- 400/401 → **not** retried, raised immediately.
- `Retry-After: 2` header → the custom wait returns ~2s (capped at `retry_max_wait_seconds`); HTTP-date form parsed too.
- Persistent 503 → exhausts after `retry_attempts` and raises (→ caller maps to `UpstreamUnavailable`).

**NIM reranker:**
- Mock transport returns 429 twice then a valid ranking → final ranking returned (proves the 429 gap in `nim_rerank.py:62` is closed).
- Persistent 429 → raises after `retry_attempts` (no infinite loop; `Retry-After` respected between attempts).

**HybridRetriever degradation:**
- Fake reranker raising `UpstreamUnavailable` → returns RRF-fused order truncated to `rerank_top_n`.
- Fake sparse raising → dense-only results returned.
- Fake embedder raising → sparse-only results returned (BM25 runs on text).
- Both dense and sparse empty/raising → `UpstreamUnavailable(stage="retrieval")` raised.
- **ACL under degradation:** org-A query with a failing reranker returns only org-A chunks (real ACL predicate, poisoned B chunk excluded).
- `degraded_stages` metadata is populated for each fallback taken.

**Store timeouts / pool:**
- Qdrant store constructed with the interactive timeout passed through (assert the client receives a non-`None` timeout).
- pgvector: a fake pool records that `register_vector` runs in the configure hook once per connection and `CREATE EXTENSION` is **not** issued on checkout (only in `ensure_collection`).
- Simulated `QueryCanceled` (statement_timeout) → surfaces as `UpstreamUnavailable`, not a 500.

**App layer:**
- `UpstreamUnavailable` from the pipeline → HTTP **503** with `{ detail, request_id }` and a `Retry-After` header when carried.
- An unexpected `RuntimeError` → HTTP **500** with `{ "detail": "Internal error", "request_id": ... }` and **no** exception text in the body.
- `X-Request-Id: abc-123` echoed back on the response; a malformed/oversized inbound id is replaced by a generated one; every response carries an `X-Request-Id`.

**Generator budget:**
- Interactive generator built with `gen_query_timeout_seconds`/`gen_query_max_retries` (assert the SDK client receives them), while the ingest path still gets 600/5.

---

## 10. Files

**Create:**
- `core/resilience.py` — `is_retryable`, `retry_after_wait`, `resilient_retry`, `UpstreamUnavailable`.
- `app/errors.py` — `install_error_handlers(app)` + `request_id` middleware.
- `tests/test_resilience.py` — classification, backoff, `Retry-After`, exhaustion.
- `tests/test_hybrid_degradation.py` — per-stage fallbacks + ACL-under-degradation.
- `tests/test_api_errors.py` — 503/500 mapping, `request_id` handling.

**Modify:**
- `providers/rerankers/nim_rerank.py` — use `resilient_retry`, interactive timeout, injected attempts.
- `providers/vectorstores/qdrant_store.py` — explicit client `timeout`, `resilient_retry` on `search`/`count`.
- `providers/vectorstores/pgvector_store.py` — `psycopg_pool` pool, configure hook (`register_vector` + `statement_timeout`), `CREATE EXTENSION` → `ensure_collection`, retry wrap.
- `providers/generators/openai_compatible.py` — accept an interactive timeout/max-retries band (keep ingest default).
- `retrieval/hybrid.py` — per-stage try/except degradation + `degraded_stages`.
- `core/config.py` — new knobs (§7).
- `core/registry.py` — pass interactive timeout/attempts into reranker + stores.
- `app/api.py` — call `install_error_handlers(app)`; surface `degraded_stages` in the response metadata.
- `core/pipeline.py` — propagate `degraded_stages` into `Answer.metadata` for observability.
- `pyproject.toml` — add `psycopg_pool` (tenacity/httpx already present).

---

## 11. Open questions / future hooks

- **Circuit breaker (optional).** A per-provider breaker (open after N consecutive `UpstreamUnavailable`, short-circuit to the degraded path for a cooldown) would cut latency during a sustained outage. Proposed as a swappable, registry-wired component gated by `circuit_breaker_enabled` — deferred so this slice stays focused; degradation already prevents a cascade for the common cases.
- **Async migration.** Making `/query` and the providers `async` (httpx `AsyncClient`, async psycopg pool, `run_in_threadpool` only for CPU-bound local rerank) would remove the fixed 40-thread ceiling entirely. Bigger change; the timeout/retry policy here is the prerequisite and stays valid either way.
- **Retry-After propagation to the client** across multiple stages: which stage's `Retry-After` wins when several degrade? Current choice: the one that actually causes the 503 (the retrieval hard-fail). Revisit if a breaker lands.
- **Per-tenant fairness under overload.** Bounded retries help globally but do not isolate a noisy tenant; pairs naturally with SP7/SP9 rate-limiting.
- **Qdrant/psycopg exception surface.** The exact wrapping types (`UnexpectedResponse`, `OperationalError`, `QueryCanceled`) should be pinned against the installed client versions during implementation; the classifier in `core/resilience.py` centralizes any adjustment.
