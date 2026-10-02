# Audit — App Layer, Observability & Deployment

**Scope:** `app/*` (API, documents router, UI/console), `observability/*` (Langfuse tracing, cost, dashboard), deployment surface (`Dockerfile`, `infra/docker-compose.yml`, `.dockerignore`, `Makefile`, `pyproject.toml`, `scripts/*`, CI in `.github/workflows/eval-gate.yml`).
**Method:** Read-only inspection of source with file:line evidence; claims cross-checked against `docs/PROJECT_STATUS.md` and the (untracked) `docs/PRODUCTION_READINESS_AUDIT.md`. External behaviors (GitHub Actions context rules) verified against upstream issue trackers. No code was executed, edited, or installed.
**Date:** 2026-08-24

---

## Verdict

| Area | Grade | One-line |
|---|---|---|
| App/API layer | **B−** | Auth model and input validation are genuinely solid; operational hardening (error mapping, concurrency bounds, readiness) is missing |
| Observability | **B−** | Thoughtful, privacy-conscious Langfuse integration that is silently lossy in several ways (init failures swallowed, native usage/cost fields never populated, no flush) |
| Deployment | **C** | Baked image exists and is well-built, but the containerized API cannot serve traffic as shipped, there is no rate limiting, and **CI is structurally broken** |
| Secrets hygiene | **C−** | A real NVIDIA key sits in an untracked, unignored `.env.bak-*` file that both `git add -A` and the Docker build would capture |

**Three things that matter most (P0):**
1. **CI is invalid as written** — `eval-gate.yml:116` uses the `secrets` context in a *job-level* `if:`, which GitHub Actions rejects at parse time (`Unrecognized named-value: 'secrets'`). The entire workflow — including the lint/offline-test job — fails to run. There is currently **no functioning CI at all**, which also means the celebrated eval gate protects nothing even before the known missing-baseline problem.
2. **Live credential in `infra/.env.bak-1785181911`** — a real 70-char `nvapi-` key (verified format/length; value not reproduced here) in a file matched by neither `.gitignore` (`*.env`, `.env.local` only) nor `.dockerignore` (`.env`, `.env.local`, `infra/.env` only). `COPY . .` in the Dockerfile would bake it into every image; `git add -A` would stage it. `PROJECT_STATUS.md` §8 explicitly claims this leak class is closed — true for `infra/.env`, already false again via the backup file created 2026-07-28. Rotate the key; delete the backup; widen ignore patterns; add a secret scanner.
3. **The containerized `api` service cannot work as shipped** — its compose environment carries only `REDIS_URL/QDRANT_URL/PG_DSN/DOC_REGISTRY_BACKEND`; there is no `env_file:` and (by design) no secrets in the image. The app therefore boots with an empty `JWT_SECRET` (every request 401s at the verifier) and empty provider keys (first embed/generate call fails). Only `ingest-worker` functions, because it bind-mounts the repo and inherits `.env` through pydantic-settings.

---

## Strengths (verified in code)

**App layer**
- **Identity strictly from verified JWT** (`app/api.py:59-63`, `app/auth.py:35-53`): `QueryRequest` carries no identity field; missing/invalid bearer → 401 with `WWW-Authenticate`. The old "client-controlled tenant" hole is genuinely closed.
- **Algorithm pinning defeats alg-confusion**: `providers/auth/jwt_verifier.py` pins the algorithm from config, never from the token header; RS256/JWKS path exists for prod.
- **Dev-console gating is defense-in-depth**: `/ui` and `/ui/token` 404 unless `auth_dev_signer_enabled` *and* `JWT_SECRET` (`app/ui.py:35-44`), and `_validate_auth` (`core/config.py:231-243`) refuses to boot with the flag set when `APP_ENV=prod`.
- **Console XSS discipline**: `console.html` renders every server-derived string via `textContent` through a single `el()` helper (lines 191-201); no `innerHTML` anywhere; retrieved document text is treated as inert.
- **Upload path validates early and scopes storage**: content-type allowlist → 415 before body read, size guard → 413 (`app/documents.py:130-145`); blob keys hash the tenant segment so hostile tenant IDs can't traverse (`_blob_key`, lines 106-110); status reads are tenant-scoped; `collection_id` rejects control characters.
- **Boot-time config validation exists** (4 `@model_validator`s in `core/config.py`, e.g. prod must have issuer/audience and must not enable the dev signer; PII-hash salt enforcement).

**Observability**
- **Privacy-first trace design**: input-guard runs and applies redactions *before* the root span is created; blocked queries trace as `[BLOCKED]` with reason, never the payload (`core/pipeline.py:112-140`); the Langfuse client is constructed with a recursive PII `mask=` callback that fails closed to `[PII_REDACTION_ERROR]` (`observability/langfuse_tracing.py:129-155`); tenant tagged on the root span.
- **Opt-in and dependency-clean**: lazy `langfuse` import only when enabled; fully functional no-op fallback (`Tracer`/`_NoOpSpan`), so tests/offline runs never touch the package; `sample_rate` is configurable.
- **Spans cover every pipeline stage** (guardrail.input, rewrite, retrieval, generation, guardrail.output — `core/pipeline.py:120-232`) including indirect-injection flags on retrieved content.
- **Cost accounting has honest edges**: `cost.py` warns once per unknown model instead of silently pricing $0, and prices are labeled estimates; `dashboard.py` is a clean offline aggregator.

**Deployment artifacts**
- **Dockerfile fundamentals are right**: dependency layer cached before source copy (`uv sync --frozen` twice, `pyproject.toml`+`uv.lock` first), locked installs, non-root user, single image serves API and worker via command override.
- **`.dockerignore` is unusually thorough** (secrets, data, caches, tests, docs, agent scratch) — apart from the `.bak` gap above.
- **Compose is honestly labeled DEV-ONLY** with per-service comments explaining the (deliberate) dev patterns (bind-mount worker, baked api, shared cache volume), and Langfuse v3 stack is a faithful upstream-style layout with healthchecks on db/clickhouse/minio/redis.
- **CI *design* is strong on paper**: offline lint/tests, live-store ACL-isolation job against service containers with a sensible host-side Qdrant `/readyz` poll (the image ships no curl), eval gate with NaN-cannot-pass semantics, and a final aggregate status gate with a labeled `eval-skip-approved` escape hatch for fork PRs.

---

## Defects & risks

Severity: 🔴 critical · 🟠 high · 🟡 medium · ⚪ low. Every finding was verified against current working-tree code.

### A. App / API layer

**A1. 🟠 No rate limiting, no concurrency cap, no app-level deadline — cheap denial-of-wallet and threadpool exhaustion.**
`app/api.py` registers zero middleware (no limiter, no CORS/TrustedHost, no timeout wrapper). The only input bound is `max_question_chars=8000` (`core/config.py:127`). Each `/query` performs blocking embed → Qdrant → BM25 → rerank → LLM work with a **600 s** upstream ceiling and 5 retries (`core/config.py:87-88`). The handler is a sync `def`, so requests run on Starlette's default anyio threadpool (~40 threads): ~40 concurrent slow queries pin the entire service, and every request is unmetered paid inference. Note the README's "100× scale" table claims rate limiting "moves from the per-instance question **cap**" to Redis — implying an existing cap; none exists beyond question length. *Fix:* gateway or middleware token bucket, a per-request deadline well under 600 s, and a bounded semaphore around pipeline calls.

**A2. 🟠 Unhandled pipeline exceptions return raw 500s.**
`app/api.py:92` calls `pipeline.run(...)` with no try/except and the app registers no exception handler. Any provider hiccup (401 from NIM on a missing key, Qdrant reset, judge timeout) surfaces as an opaque HTTP 500 with no correlation ID, no structured body, no distinction between transient-upstream vs bug. *Fix:* map known failure classes to 502/504/400, log server-side with a request ID, sanitize the client response.

**A3. 🟡 `/healthz` is liveness-only — there is no readiness signal anywhere.**
`app/api.py:74-76` returns a static `{"status": "ok"}` without touching Qdrant/Postgres/Redis/pipeline. Combined with the lazy pipeline build (A4), an instance whose vector store is unreachable looks perfectly healthy to any orchestrator/load balancer. There is also no `HEALTHCHECK` in the Dockerfile, so even `docker run` has no probe. *Fix:* add `/readyz` that pings dependencies (and reports tracing health), wire it into compose `healthcheck:`.

**A4. 🟡 Lazy pipeline singleton has an unguarded check-then-set race.**
`get_pipeline()` (`app/api.py:27-34`) builds the heavy RAGPipeline (models, stores) on first request with no lock while handlers execute in threads — concurrent cold-start requests can each trigger a full build (memory/connection blowup), last write wins. *Fix:* build in a FastAPI lifespan startup hook (which also makes A3's readiness meaningful) or double-checked locking.

**A5. 🟡 Async ingest can strand documents in `processing`/`deleting` forever.**
Upload order is blob-put → registry-create(`PROCESSING`) → enqueue (`app/documents.py:147-163`); delete sets `DELETING` then enqueues (`documents.py:223-224`). If the enqueue fails (Redis down) or the arq worker crashes mid-job — `WorkerSettings` (`ingest/worker.py:166-182`) configures **no retries, no job_timeout, no max_tries**, and arq does not redeliver lost jobs — the row never leaves its transitional state and nothing sweeps it. Blob writes are also not compensated if registry create fails. *Fix:* outbox/reconciler sweep for stale transitions; explicit `job_timeout` + retry policy; wrap enqueue failure to mark `failed`.

**A6. ⚪ Minor API nits.**
`list_documents` is unpaginated (returns every doc per tenant). Upload enforces size *after* reading the whole body into memory (`raw = await file.read()`, `documents.py:138`) — acceptable at 25 MiB but worth streaming later. `make api` documents `uvicorn --reload` as the launch command (Makefile:36) — dev-server-as-documented-run-path; no production target exists.

### B. Observability

**B1. 🟡 Langfuse dashboards see zero tokens/cost — native fields never populated.**
All observations are opened as generic spans (`start_as_current_observation(as_type="span")`, `langfuse_tracing.py:186-188`); generation tokens go into free-form `metadata` (`core/pipeline.py:207-213`) and cost into `root.update(output={"cost_usd": ...})` (`pipeline.py:254-261`). Langfuse only aggregates token/cost through `usage_details=`/`cost_details=` on generation-type observations, so the platform's model/cost analytics stay empty even when tracing is fully enabled — the deliverable "LLM token counts and estimated cost" lands only as opaque JSON blobs. *Fix:* open the generation stage as `as_type="generation"` with `model=ans.model` and pass structured usage/cost fields.

**B2. 🟡 Tracing failure is indistinguishable from tracing-off.**
Every Langfuse interaction swallows exceptions silently: client init failure flips `_enabled=False` with no log (`langfuse_tracing.py:132-134`); span creation falls back to no-op (`185-191`); update/score/flush are bare `except: pass`. Graceful degradation is the right *behavior*, but a wrong host or expired keys produces total silence — operators believe traces flow when none do. *Fix:* warn-once logging on init failure and repeated span errors; surface tracer status in `/readyz`.

**B3. 🟡 Anthropic path is priced at $0.00.**
`core/pipeline.py:249-253` computes `cost_usd(self.settings.gen_model, ...)`. When `GEN_PROVIDER=anthropic` the generator serves `anthropic_model` (`claude-sonnet-4-6`) but `gen_model` still holds the NIM id `meta/llama-3.3-70b-instruct`, whose PRICING entry is `(0.0, 0.0)` — real spend is under-reported as free. This is self-documented in `PROJECT_STATUS.md` §7 but still unfixed in code. *Fix:* price against the server-returned model id (`ans.model`), falling back to settings only when unknown.

**B4. 🟡 No flush at process boundaries.**
`Tracer.flush()` exists (`langfuse_tracing.py:257-263`) but has no production caller; the FastAPI app registers no lifespan/shutdown hook. The v4 SDK batches (~5 s interval) and registers its own atexit, so clean exits mostly survive — but SIGKILL/scale-in drops the last window of spans, exactly the ones you want during incidents. *Fix:* lifespan shutdown hook calling flush/shutdown; flush at eval/CLI exit points.

**B5. ⚪ Dead documented surface:** `Tracer.log_query_trace()` (`langfuse_tracing.py:200-251`) — presented in the module's public-surface docs — has no production caller (tests only); retrieval hit IDs/scores never reach spans either (only `n_hits`, `pipeline.py:191`), so Langfuse can't answer "why did this query retrieve wrong chunks".

**B6. ⚪ Naive p95:** `dashboard.py:69` computes `p95_idx = max(0, int(n*0.95)-1)` — for n≤20 it systematically mis-reports (n=2 returns the *min* as p95), no interpolation. Use `statistics.quantiles`.

### C. Deployment, infra & CI

**C1. 🔴 The CI workflow is invalid — no job runs at all.**
`.github/workflows/eval-gate.yml:116` conditions the eval job with `if: github.repository == '…' && secrets.NVIDIA_API_KEY != ''`. The `secrets` context is **not available in `jobs.<job_id>.if`** (GitHub docs context-availability table; long-standing runner issue actions/runner#520): workflow parse fails with `Unrecognized named-value: 'secrets'`, rejecting the *whole file*. Consequence: lint, ACL-isolation, eval, and the status gate never execute — the repo currently has **no CI whatsoever**, which is worse than "gate without a baseline" and contradicts PROJECT_STATUS §2's "📋 CI … awaiting implementation" framing (it's not awaiting, it's broken). *Fix:* drop the secrets clause from the job-level `if` (keep only `github.repository ==`), or gate inside steps / via a prior check-secrets job emitting an output; validate locally with `actionlint`.

**C2. 🔴 Live API key in an unignored, undockerignored backup file.**
`infra/.env.bak-1785181911` contains a real 70-char `nvapi-` key (verified length/format; deliberately not reproduced here — note that `PRODUCTION_READINESS_AUDIT.md` also prints a full key verbatim in its own finding text, so that on-disk doc is itself a leak artifact). `.gitignore` covers `.env`/`.env.local`; `.dockerignore` covers `.env`, `.env.local`, `infra/.env` — **neither matches `*.bak*`**, so one `git add -A` or one `docker build` bakes the key into image layers. PROJECT_STATUS §8 (line ~225) asserts this exact class is handled ("gitignored … excluded from the Docker build context") — true for `infra/.env`, already false again via this file dated 2026-07-28. *Fix:* rotate the key now; delete the backup; ignore `infra/.env*` and `**/*.env.*`; add gitleaks to pre-commit and CI.

**C3. 🟠 The baked `api` container cannot serve traffic as shipped.**
The compose `api` service (`infra/docker-compose.yml:50-70`) sets only `REDIS_URL/QDRANT_URL/PG_DSN/DOC_REGISTRY_BACKEND` — no `env_file:`, and the image intentionally contains no `.env`. So inside the container: `JWT_SECRET=""` → HS256 verification fails → **every request 401s**; provider keys empty → even authenticated requests would fail at the first embed/generate call; Langfuse keys absent → tracing off regardless of intent. Only `ingest-worker` works, because it bind-mounts the repo (`../:/workspace`) and pydantic-settings reads `.env` from the working dir — an asymmetry that makes `make up`'s headline "everything" misleading. *Fix:* add `env_file: [.env]` (or compose secrets) to `api`, and document the required env contract next to the service.

**C4. 🟡 No production serving story: single uvicorn process, dev reload documented.**
Dockerfile CMD runs bare `uvicorn` (no `--workers`), no gunicorn; Makefile's only run target uses `--reload` (dev watcher). No image publishing CI, no k8s/helm, no TLS/reverse-proxy guidance — consistent with what PROJECT_STATUS §9(4) admits ("rate limiting / k8s / image publishing / proxy deferred"), but worth stating plainly: today's deployable unit is one process per container with manual scaling.

**C5. 🟡 Compose hygiene gaps (acknowledged DEV-ONLY, still worth listing).**
- `qdrant/qdrant:latest` unpinned in both compose and both CI jobs; several other images float on major tags.
- Port exposure is inconsistent: app postgres publishes `5432:5432` on all interfaces (weak rag/rag creds), Qdrant publishes `6333/6334` unauthenticated on all interfaces, langfuse-web `3000` likewise — while langfuse-db/clickhouse/minio/redis correctly bind `127.0.0.1`. On any shared network, Qdrant is world-writable.
- Redis healthcheck is `redis-cli ping` while the server requires auth — it validates liveness but not the configured credential path.
- One Redis instance serves arq queue + semantic cache **and** all Langfuse queues/cache (documented in PROJECT_STATUS §3): a Langfuse queue blowup or `maxmemory-policy` change takes down ingestion coupling-wise; `restart:` policies are mixed (`unless-stopped` vs `always`).
- No resource limits anywhere; qdrant has no healthcheck (hence `condition: service_started` in dependents).

**C6. ⚪ Makefile nits:** `.PHONY` declares a `demo` target that doesn't exist; README quickstart tells you to hand-run `arq ingest.worker.WorkerSettings` in a separate shell while there's no `make worker` target (the worker exists only as a compose service); `make console` embeds a fallback JWT secret literal.

### D. Docs & self-assessment drift

**D1. ⚪ `docs/PRODUCTION_READINESS_AUDIT.md` is a stale, untracked artifact that still shapes perception.**
It is genuinely not git-tracked (matches PROJECT_STATUS's disclaimer), but its content predates major fixes: it reports "No Dockerfile for the application" and "no config validation on boot" — both false since SP9 — and cites `app/demo.py`, which no longer exists. It also prints the live NVIDIA key verbatim (see C2). Recommendation: delete it or move it under `docs/review/` with a prominent SUPERSEDED banner, scrubbed of secrets.

**D2. ⚪ Internal inconsistencies in the current docs.**
- PROJECT_STATUS §2 says observability "leaks raw queries when on", while §8 item 4 (and the code, `core/pipeline.py:112-140`) say pre-trace redaction is fixed — §2 is stale.
- README's Status section lists "deploy packaging (Dockerfile, rate limiting)" as outstanding; the Dockerfile has landed (SP9) — half-stale.
- README "100× scale" implies a current per-instance question cap that doesn't exist (A1).
These matter because the docs are otherwise unusually honest — drift erodes the trust their accuracy earns.

---

## Prioritized recommendations

**P0 — do now (hours, unblocks everything else)**
1. Rotate the NVIDIA key exposed via `infra/.env.bak-1785181911` (and treat the one printed inside `PRODUCTION_READINESS_AUDIT.md` as exposed); delete both artifacts; add `infra/.env*`, `*.bak*` to `.gitignore`/`.dockerignore`; wire gitleaks into pre-commit + CI.
2. Fix `eval-gate.yml:116`: remove `secrets.*` from the job-level `if:` (keep `github.repository == …`), move secret-availability detection into a step or a check job with outputs; run `actionlint` in CI to prevent regressions. Only after this does the "eval gate" story become real (baseline-run work can proceed meaningfully).

**P1 — before any real traffic (days)**
3. Give compose `api` its environment: `env_file: .env` + documented required-vars contract; smoke-test `make up` → mint token → upload → query end-to-end.
4. Add `/readyz` (deps + tracer status), Dockerfile `HEALTHCHECK`, lifespan startup build of the pipeline (kills A4), and an exception handler mapping upstream failures to 502/504 with a request ID.
5. Rate limiting + per-request deadline: middleware token bucket, bounded semaphore around pipeline calls, drop the effective 600 s hot-path ceiling to something sane for interactive queries.

**P2 — hardening (weeks)**
6. Langfuse: generation-type spans with structured `usage_details`/`cost_details`; price against server-returned model id; warn-once on init/span failures; shutdown flush.
7. Ingest reliability: arq `job_timeout`/retry policy, stale-transition sweeper, enqueue-failure compensation; paginate `list_documents`.
8. Compose: pin image digests, bind all ports to `127.0.0.1` by default, auth-aware Redis healthcheck, qdrant healthcheck + restart policy, resource limits; consider splitting Langfuse's Redis from the app Redis.
9. Docs: reconcile §2 observability row and README Status; delete/supersede the old audit doc.

---

## Appendix — evidence base

- Read in full: `app/api.py`, `app/ui.py`, `app/documents.py`, `app/auth.py`, `app/static/console.html`, `observability/{langfuse_tracing,cost,dashboard}.py`, `core/config.py`, `core/pipeline.py`, `ingest/worker.py`, `Dockerfile`, `Makefile`, `pyproject.toml`, `infra/docker-compose.yml`, `infra/.env.example`, `.github/workflows/eval-gate.yml`, `README.md`, `docs/PROJECT_STATUS.md`; skimmed `docs/PRODUCTION_READINESS_AUDIT.md`; inspected `.gitignore`, `.dockerignore`, git tracking state (`git ls-files` / `check-ignore`), and byte-level content of suspect lines.
- External checks: GitHub Actions context-availability rule for `jobs.<job_id>.if` (actions/runner#520) confirming C1.
- Verification note: several credential strings in this tree appear masked in some tooling output (e.g., DSN passwords render as `***`); findings above were confirmed against raw file bytes where masking was suspected. No keys are reproduced in this report.

