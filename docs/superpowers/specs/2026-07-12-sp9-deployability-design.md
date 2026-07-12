# SP9 · Deployability & Ops — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 2. Turns the working code into a deployable, self-defending artifact. Depends on SP1 (auth boot-check + per-field input caps — extended here, not duplicated); coordinates with but excludes SP6 (hot-path client resilience + global exception handler) and SP5 (eval CI). This slice is independently implementable once SP1's `Settings` boot validator exists.

---

## 1. Context & problem

The system runs but is not deployable or hardened for ops. Every defect below was verified against the current source (line numbers re-checked; the audit hints had drifted).

| # | Defect | Location (verified) | Reality |
|---|--------|---------------------|---------|
| 1 | No app container | repo root — `Dockerfile`, `.dockerignore` absent | `ls Dockerfile .dockerignore infra/Dockerfile` → all missing. The app cannot be packaged or shipped as an image. |
| 2 | Only launch path is a dev server | `Makefile:37` | `api:` target runs `uvicorn app.api:app --reload` — single worker, auto-reload, no process manager. Not a production runtime. |
| 3 | No config validation at boot | `core/config.py:107-123` | The `_fill_key_fallbacks` validator fills keys but never *asserts* one is non-empty, and never checks `embed_dimension` against the store's configured vector size. An empty key or a dimension mismatch boots "healthy" and surfaces as an opaque 401 / garbage similarity on the first `/query`. |
| 4 | `/healthz` is not a readiness probe | `app/api.py:64-66` | `healthz()` returns `{"status": "ok"}` unconditionally — no dependency is pinged. A misconfigured instance passes the load-balancer check, receives traffic, then 500s every query. |
| 5 | No rate limit, body-size cap, app-level timeout, CORS/TrustedHost, or security headers | `app/api.py:44` (`FastAPI(...)` — zero middleware) | The most expensive path is uncapped: a modest burst of large bodies pins the sync threadpool for minutes (denial-of-wallet). No `Host`/origin allowlist; no `X-Content-Type-Options`/`X-Frame-Options`/HSTS. |
| 6 | Compose backends lack restart/healthcheck/resource limits | `infra/docker-compose.yml:17-39` | `qdrant` has **no** `restart:`, **no** `healthcheck:`, **no** resource limits (lines 17-23); `postgres` has a healthcheck (35-39) but no `restart:`/limits. Backends won't self-heal, and nothing gates on their readiness. |
| 7 | Racy lazy pipeline singleton on the sync hot path | `app/api.py:24-33`, `app/api.py:69-74` | `get_pipeline()` does an unguarded lazy `if _pipeline is None: build(...)`. Every route is a plain `def`, so FastAPI runs it in a bounded (~40-thread) threadpool; concurrent first requests race to build the heavy pipeline (double model load / double store handshake). |
| 8 | No `.dockerignore`; no secret scanner | repo root; `.github/workflows/` | Only `eval-gate.yml` exists — no gitleaks. `infra/.env` holds a live `nvapi-` key (`infra/.env:1`); a `COPY .` into an image would bundle it. Git history is clean (gitignored, never committed), so this is build-context + pre-commit hygiene, not a history rewrite. |

---

## 2. Goals

- Ship a **production Dockerfile** (multi-stage, non-root, no `--reload`) and a non-dev launch (uvicorn workers under a process manager).
- Add the **app as a compose service** with `depends_on: { condition: service_healthy }`, and give `qdrant`/`postgres`/`clickhouse` `restart:`, `healthcheck:`, and `deploy.resources.limits`.
- **Fail closed at boot**: assert the selected provider has a non-empty key, and assert `embed_dimension` matches the store's *configured* vector size (when a collection exists) — extending SP1's existing boot validator, not adding a second one.
- Build the pipeline **once, in a FastAPI lifespan handler**, eliminating the racy lazy singleton.
- Split liveness from readiness: keep `/healthz` cheap; add a **real readiness probe** that cheaply pings Qdrant/embedder without paid inference.
- Add transport-layer self-defense: a **rate limiter**, a **request-body-size cap**, an **app-level request timeout** well below the 600s SDK ceiling, **TrustedHost + CORS**, and **security headers**.
- Add a **`.dockerignore`** (excludes `infra/.env`, `.env`, `.git`, `.venv`, `data/`, `.cache/`) and a **gitleaks** pre-commit + CI scan for `nvapi-`/`sk-` patterns.
- Keep swappable concerns (rate limiter) behind the existing Protocol + registry pattern; keep cross-cutting HTTP concerns as middleware.

## 3. Non-goals (deferred) — owned elsewhere

- **Global exception handler + `request_id` correlation** → **SP6** (Resilience). SP9 sanitizes error *shapes* it introduces but does not register the catch-all 500 handler.
- **Hot-path client timeouts / retry-on-429/5xx / per-stage degradation / 503 mapping** for Qdrant, pgvector, the NIM reranker, and the generator → **SP6**. SP9's "app-level request timeout" is a single *middleware* wall-clock bound, distinct from the per-client resilience work.
- **Per-field input caps** (`max_question_chars` → 422, `max_acl_tags`) and the **auth boot-check** → already delivered by **SP1**. SP9 *extends* SP1's boot validator and adds only the *transport* body-size cap (Content-Length), never re-declaring `max_question_chars`.
- **Eval CI job / paired-bootstrap gate** → **SP5**. SP9 adds only the gitleaks CI job; it does not touch `eval-gate.yml`'s eval logic.
- **BM25 cross-worker staleness / post-ingest reload** → **SP7** (Ingest/retrieval robustness). SP9 fixes the *build race* (lifespan singleton) only; multi-worker BM25 freshness is a separate finding, noted in §11.
- **Key rotation** → operational action, not a spec artifact. SP9 delivers the `.dockerignore` + scanner so the key can't leak via build context; rotating the live key is an ops runbook item.
- **Physical per-tenant isolation / namespace-per-org** → VDB-Decision / SP11.

---

## 4. Decisions (PROPOSED)

These are proposed for the user to confirm or override on review. Each leads with the best-practice option.

| # | Decision | Proposed choice | Rationale |
|---|----------|-----------------|-----------|
| D1 | Runtime | **uvicorn with `--workers N` (no `--reload`)** under a container; gunicorn+uvicorn-worker optional if a richer process manager is wanted | Multiple workers for CPU-bound sync path; `--reload` is a dev-only footgun. Uvicorn-only keeps the dependency set minimal. |
| D2 | Image | **Multi-stage, `uv`-based, non-root user, `python:3.12-slim` base, `HEALTHCHECK` on `/readyz`** | Small, reproducible, least-privilege; matches `requires-python >=3.11,<3.14`. |
| D3 | Liveness vs readiness | **`/healthz` stays cheap/unconditional (liveness); add `/readyz` as the dependency-pinging readiness probe** | A liveness probe that pings deps causes restart-loops when a dep blips. Kubernetes-idiomatic split. |
| D4 | Readiness cost | **Readiness uses only *non-paid* signals**: Qdrant `get_collections()` + a cached boot-time embedder-connectivity result (`models.list()` at boot), never a live `embed_query` per probe | A paid embed call per probe is denial-of-wallet via health checks. |
| D5 | Boot config validation | **Extend SP1's `Settings` boot check**: assert active provider key non-empty; assert `embed_dimension == store.configured_dimension()` when a collection exists (warn+skip when absent) | One boot path; fail closed on misconfig; nothing to compare pre-ingest. |
| D6 | Pipeline lifecycle | **Build once in a FastAPI `lifespan` handler; store on `app.state`; `get_pipeline` reads it** | Removes the race; no per-request lazy build; deterministic startup failure. |
| D7 | Rate limiter | **Swappable `RateLimiter` Protocol; in-memory token-bucket default, Redis-backed impl as a future drop-in; wired in `core/registry.py`** | Genuinely swappable → belongs in the registry, consistent with every other component. |
| D8 | Cross-cutting HTTP concerns | **FastAPI middleware** for CORS, TrustedHost, security headers, body-size cap, request timeout — config-knob driven, *not* Protocols | These are framework middleware, not swappable domain components; forcing them into the registry invents structure. |
| D9 | Dimension introspection | **Add `VectorStore.configured_dimension() -> int | None` to the Protocol**, implemented for Qdrant + pgvector | Keeps boot/readiness from reaching into client internals; `None` cleanly encodes "collection absent". |
| D10 | Body-size cap | **Reject on `Content-Length` before JSON parse** (middleware), returning `413` | Stops oversized bodies from being buffered/parsed; cheap and early. |
| D11 | App request timeout | **New `request_timeout_ms` middleware bound, default 45s**, independent of the 600s SDK `request_timeout_seconds` | Bounds a single request's wall-clock so a stuck handler frees its thread; the SDK ceiling stays generous for legit slow inference. |
| D12 | Secret scanning | **gitleaks as a pre-commit hook *and* a standalone CI job** scanning for `nvapi-`/`sk-` patterns | Defense in depth: catch before commit and on every push. |
| D13 | Qdrant compose healthcheck | **TCP/bash readiness check** (`bash -c ':> /dev/tcp/127.0.0.1/6333'` or qdrant's documented probe), not `curl` | The `qdrant/qdrant` image ships without curl/wget; a curl healthcheck would never pass. |

---

## 5. Architecture & components

Following the codebase pattern: Protocols in `core/interfaces.py`, concrete impls in `providers/`, wired only in `core/registry.py`. Cross-cutting HTTP concerns are middleware, not Protocols.

### 5.1 `RateLimiter` (Protocol) — `core/interfaces.py`
```
@runtime_checkable
class RateLimiter(Protocol):
    def allow(self, key: str) -> bool: ...   # False → over limit (→ 429)
```
- `providers/ratelimit/token_bucket.py` — `InMemoryTokenBucket(rate, burst)`: per-key token bucket, monotonic-clock refill, thread-safe (a `Lock` — the sync threadpool means real concurrency). Default impl.
- `providers/ratelimit/redis_bucket.py` — future `RedisTokenBucket` for multi-worker/multi-instance correctness (a per-process in-memory limiter is per-worker; documented limitation). Not built in this slice; the Protocol makes it a drop-in.
- Wired via `core/registry.py::build_rate_limiter(settings)` — the only place the concrete class is named. Keyed by `Principal.tenant_id` (from SP1) so the limit is per-tenant, falling back to client host when auth is disabled in dev.

### 5.2 `VectorStore.configured_dimension()` — `core/interfaces.py` (+ impls)
Add to the `VectorStore` Protocol:
```
def configured_dimension(self) -> int | None: ...   # None when collection absent
```
- `QdrantVectorStore`: read `get_collection(collection).config.params.vectors.size`; `None` if the collection isn't in `get_collections()`.
- `PgVectorStore`: introspect the column typmod / `information_schema` for the vector dimension; `None` if the table is absent.
Used by the boot check (D5) and `/readyz` (D4).

### 5.3 Boot validation — `core/config.py` + `app/lifespan.py`
- **Static checks** (extend SP1's `Settings` boot validator): the active provider's key (per `gen_provider`/`vector_store`/embedder base-url) is non-empty; conservative range checks on the new knobs. Pure, offline-testable.
- **Live checks** (in the lifespan handler, since they touch I/O): call `store.configured_dimension()`; if it returns an int and it ≠ `embed_dimension`, **raise → refuse to start** (fail closed). If `None` (pre-ingest), log a warning and continue. Ping the embedder once (`models.list()`) and cache the boolean for `/readyz`; a failure here refuses to start on a *served* instance.

### 5.4 Lifespan pipeline builder — `app/lifespan.py`
An `asynccontextmanager` `lifespan(app)`:
1. Load `settings`; run static + live boot validation (§5.3).
2. `app.state.pipeline = build(version="full", dataset=None)` — once.
3. `app.state.rate_limiter = build_rate_limiter(settings)`.
4. `app.state.ready = True` on success.
`get_pipeline()` becomes a dependency that returns `request.app.state.pipeline` (no lazy build, no race). Wired via `FastAPI(lifespan=lifespan)`.

### 5.5 Middleware stack — `app/middleware.py`
Registered on `app` (outer→inner order matters):
- `TrustedHostMiddleware(allowed_hosts=settings.allowed_hosts)` — reject unknown `Host`.
- `CORSMiddleware(allow_origins=settings.cors_allow_origins, ...)` — deny-by-default; empty list = no cross-origin.
- `SecurityHeadersMiddleware` (custom) — sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and HSTS when `settings.hsts_enabled`.
- `BodySizeLimitMiddleware` — reject when `Content-Length > settings.max_body_bytes` → `413`, before parsing.
- `RequestTimeoutMiddleware` — wrap the handler in `asyncio.wait_for(..., settings.request_timeout_ms/1000)`; on timeout return `503` (a bounded, sanitized shape).
- Rate-limit check (dependency or middleware): `if not rate_limiter.allow(key): 429` with `Retry-After`.

### 5.6 Container + compose — `Dockerfile`, `.dockerignore`, `infra/docker-compose.yml`
- `Dockerfile`: multi-stage (`uv sync --extra app` build stage → slim runtime), non-root `appuser`, `CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000","--workers","4"]`, `HEALTHCHECK` hitting `/readyz`.
- `.dockerignore`: excludes `infra/.env`, `.env*`, `.git`, `.venv`, `data/`, `.cache/`, `eval/runs/`, `*.pkl`, `.pytest_cache`, `.ruff_cache`, `.remember/`, `.claude/`.
- compose: add an `app` service (`build: .`, `depends_on: { qdrant: {condition: service_healthy}, postgres: {condition: service_healthy} }`, `restart: unless-stopped`, resource limits); add `restart:` + `healthcheck:` + `deploy.resources.limits` to `qdrant`, `postgres`, `clickhouse`.

### 5.7 Secret scanning — `.gitleaks.toml`, `.pre-commit-config.yaml`, `.github/workflows/gitleaks.yml`
- gitleaks config with an extra rule for `nvapi-[A-Za-z0-9_-]+` (plus the built-in `sk-` OpenAI rule).
- pre-commit hook + a CI workflow job that fails the build on any finding.

---

## 6. Data flow

```
Container start
  uvicorn --workers N  → each worker process:
    FastAPI(lifespan):
      1. load settings
      2. STATIC boot validation (SP1 validator + SP9 key/knob checks)     ── fail → process exits
      3. LIVE boot validation: store.configured_dimension() vs embed_dimension,
         embedder models.list() ping                                       ── mismatch/unreachable → process exits (served)
      4. build pipeline ONCE → app.state.pipeline
      5. build rate limiter  → app.state.rate_limiter
      6. app.state.ready = True

Request  POST /query
  TrustedHost → CORS → SecurityHeaders → BodySizeLimit(413) → RequestTimeout(503)
    → rate_limiter.allow(tenant_key)?  no → 429 (+Retry-After)
    → require_principal (SP1)  → Principal
    → pipeline = app.state.pipeline   (no lazy build, no race)
    → pipeline.run(question, acl)  → QueryResponse

Probe  GET /healthz  → {"status":"ok"}                 (cheap, always; liveness)
Probe  GET /readyz   → app.state.ready
                       AND qdrant get_collections() ok
                       AND cached embedder-connectivity flag
                       → 200 ready / 503 not-ready      (readiness)
```

---

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|------|---------|---------|
| `app_env` | `"dev"` | (from SP1) `dev`\|`prod`; gates served-instance boot checks (embedder ping, dimension mismatch → hard fail). |
| `allowed_hosts` | `["*"]` in dev, must be set in prod | TrustedHost allowlist; `*` refused when `app_env=prod`. |
| `cors_allow_origins` | `[]` | CORS origin allowlist; empty = no cross-origin (deny by default). |
| `hsts_enabled` | `False` | Emit `Strict-Transport-Security` (enable behind TLS). |
| `max_body_bytes` | `65536` (64 KiB) | Reject larger request bodies with `413` before parsing. |
| `request_timeout_ms` | `45000` | App-level per-request wall-clock bound (middleware) → `503`. Independent of the 600s SDK ceiling. |
| `rate_limit_per_minute` | `60` | Token-bucket refill rate per key (tenant). |
| `rate_limit_burst` | `20` | Token-bucket burst capacity. |
| `rate_limiter` | `"memory"` | `memory`\|`redis`; selects the `RateLimiter` impl in the registry. |
| `readiness_check_embedder` | `True` | Include the cached embedder-connectivity flag in `/readyz` (and ping at boot). |

Boot validation additions (fail closed on a served instance): active provider key non-empty; `allowed_hosts != ["*"]` when `app_env=prod`; `embed_dimension` matches `store.configured_dimension()` when a collection exists.

---

## 8. Error handling

Security-critical paths fail closed; misconfiguration fails at boot, not on the first request.

- **Boot: empty active-provider key** → raise in the static validator → process exits non-zero. Never boots "healthy" into an opaque runtime 401.
- **Boot: dimension mismatch** (`configured_dimension()` is an int ≠ `embed_dimension`) → raise in lifespan → exit. **Collection absent** (`None`) → warn + continue (nothing to compare pre-ingest; not a security failure).
- **Boot: embedder unreachable** on a served instance (`app_env=prod`, `readiness_check_embedder=True`) → exit. In dev, log a warning and mark readiness degraded rather than crash the dev loop.
- **`/readyz`**: any failing dependency → `503` with a minimal `{"ready": false, "checks": {...}}` body (no vendor error text, no stack trace). `/healthz` never depends on external state.
- **Body too large** → `413` before JSON parse (no partial buffering of the oversized payload).
- **Request timeout** → `503` (bounded, sanitized). The stuck handler's thread is freed; the abandoned work is a documented tradeoff (mirrors SP2's groundedness-timeout stance).
- **Rate limited** → `429` + `Retry-After`. Fail *closed* on limiter *internal* error: if the limiter itself raises, treat as over-limit (deny) rather than allow — never fail open into denial-of-wallet.
- **Unknown `Host` / disallowed origin** → rejected by TrustedHost/CORS (`400`/no CORS headers). Deny by default.
- **Rate-limit key when auth disabled (dev)** → fall back to client host; documented that per-process in-memory limiting is per-worker until the Redis impl lands.
- The catch-all sanitizer for *unexpected* 500s + `request_id` is **SP6**; SP9's own error responses are already sanitized shapes.

---

## 9. Testing (TDD) — offline, deterministic

Written red-first; no network, no live backends.

**Boot validation**
- Empty active-provider key → boot validator raises (parametrized over `gen_provider`/embedder).
- `configured_dimension()` returns a mismatching int → lifespan raises. Returns matching int → boots. Returns `None` → boots with a warning (assert the warning, no raise).
- `allowed_hosts=["*"]` with `app_env=prod` → refuse to start; with `dev` → allowed.

**Pipeline lifecycle (race fix)**
- With `lifespan`, `app.state.pipeline` is built exactly once (patch `build` with a call counter; drive N concurrent requests via `TestClient`/threads; assert counter == 1). Contrast: the old lazy `get_pipeline` under concurrency could exceed 1 (regression guard).

**Readiness / liveness split**
- `/healthz` → `200 {"status":"ok"}` even when the (faked) store is down.
- `/readyz` → `200` when the store fake reports collections OK and the embedder flag is `True`; `503` when the store fake raises; `503` when the embedder flag is `False`. Assert `/readyz` never calls `embed_query` (spy on the embedder — zero paid calls).

**Middleware**
- Body over `max_body_bytes` → `413`, and the JSON parser is never reached (assert via a spy/oversized Content-Length).
- Handler exceeding `request_timeout_ms` (sleep fake) → `503`.
- Rate limiter: N+1th call within the window → `429` + `Retry-After`; limiter raising internally → `429` (fail closed), not `200`.
- Security headers present on every response; TrustedHost rejects an off-allowlist `Host`; CORS denies a non-allowlisted origin.

**Rate limiter unit**
- `InMemoryTokenBucket`: burst then refill over monotonic time (inject a fake clock — deterministic, no `sleep`); per-key isolation (tenant A's spend doesn't affect tenant B); thread-safety under concurrent `allow()`.

**Config**
- New knobs load from env with correct defaults; `rate_limiter="redis"` without the impl raises a clear `ValueError` in the registry (not an import crash at module load).

**Container / scanner (lint-level, no docker daemon required in unit CI)**
- `.dockerignore` contains `infra/.env`, `.env`, `.git`, `.venv`, `data/`, `.cache/` (string assertions on the file).
- gitleaks config matches an `nvapi-` sample and an `sk-` sample; a clean fixture yields no finding.
- compose lint: `app`, `qdrant`, `postgres`, `clickhouse` each have `restart:` and a `healthcheck:`; `app.depends_on` uses `condition: service_healthy` (parse the YAML, assert keys — no daemon needed).

---

## 10. Files

**Create**
- `Dockerfile`
- `.dockerignore`
- `app/lifespan.py` — lifespan handler: boot validation + pipeline/rate-limiter build.
- `app/middleware.py` — TrustedHost/CORS/security-headers/body-size/timeout wiring + custom middleware classes.
- `providers/ratelimit/__init__.py`
- `providers/ratelimit/token_bucket.py` — `InMemoryTokenBucket`.
- `.gitleaks.toml` — `nvapi-`/`sk-` rules.
- `.pre-commit-config.yaml` — gitleaks hook.
- `.github/workflows/gitleaks.yml` — CI secret scan.
- `tests/test_boot_validation.py`
- `tests/test_lifespan_singleton.py`
- `tests/test_health_readiness.py`
- `tests/test_middleware.py`
- `tests/test_rate_limiter.py`
- `tests/test_deploy_artifacts.py` — `.dockerignore` / gitleaks / compose-shape assertions.

**Modify**
- `core/interfaces.py` — add `RateLimiter` Protocol; add `configured_dimension()` to `VectorStore`.
- `providers/vectorstores/qdrant_store.py` — implement `configured_dimension()`.
- `providers/vectorstores/pgvector_store.py` — implement `configured_dimension()`.
- `core/registry.py` — `build_rate_limiter(settings)`.
- `core/config.py` — new knobs (§7); extend the SP1 boot validator with key + host + dimension-static checks.
- `app/api.py` — `FastAPI(lifespan=...)`; replace the lazy `get_pipeline` singleton with an `app.state` reader; register middleware; add `/readyz`; keep `/healthz` cheap.
- `infra/docker-compose.yml` — add the `app` service; add `restart:`/`healthcheck:`/resource limits to `qdrant`/`postgres`/`clickhouse`.
- `Makefile` — replace `api:` `--reload` with the container/worker launch (`up-app` gains the `app` service; a `run-api` target uses `--workers`, no reload).
- `pyproject.toml` — add `gitleaks`/`pre-commit` to a `dev`-adjacent extra if driven via Python; the `app` extra already carries `fastapi`/`uvicorn`.

---

## 11. Open questions / future hooks

- **Redis-backed rate limiter** — the in-memory token bucket is per-worker/per-process, so N workers permit up to N× the intended rate. `RedisTokenBucket` (D7, already Protocol-shaped) is the correct multi-instance fix; defer until a real multi-worker deployment exists. **Confirm the default worker count** (D1: `--workers 4`?) — it trades throughput against the per-worker limiter skew.
- **BM25 cross-worker staleness** — building the pipeline per worker in lifespan fixes the *race* but each worker holds its own in-memory BM25 index; a post-ingest reload across workers is **SP7**, not fixed here.
- **gunicorn vs uvicorn workers** (D1) — uvicorn `--workers` is simplest; adopt gunicorn+uvicorn-worker only if graceful-reload / richer worker management is required. **User to confirm.**
- **Readiness embedder signal** (D4) — `models.list()` at boot + cached flag avoids paid probes but goes stale if the provider degrades mid-run. A periodic cheap re-check could refresh it; deferred to avoid a background-task dependency here.
- **HSTS / TLS termination** — assumed handled by an upstream proxy; `hsts_enabled` defaults off. Confirm the deployment terminates TLS before the app.
- **Timeout on a running future** (D11/§8) — the middleware timeout frees the *request* but cannot cancel an in-flight blocking call in the threadpool (same limitation SP2 documents). True cancellation needs an async hot path — out of scope.
