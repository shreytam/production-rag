# Production RAG — Project Status

_Last updated: 2026-07-24. A cold-start orientation: what this project is, what's built, the tech stack, and what still needs fixing before it's production-grade. Read this first if you've been away from the codebase._

---

## 1. What this project is

A **production-grade Retrieval-Augmented Generation (RAG) system** built behind clean, swappable interfaces (no framework lock-in in the core). The thesis: most RAG demos glue an LLM to a vector DB and stop; this one adds the engineering discipline that comes *before* production — hybrid retrieval, reranking, contextual chunking, multi-tenant ACL isolation, guardrails, and a **rigorous eval harness with bootstrap confidence intervals wired into a CI gate**.

It's a portfolio/foundation project, designed to be the reusable base for two later projects (Graph RAG, Multimodal RAG), so the core pipeline sits behind Python `Protocol` interfaces and **`core/registry.py` is the only module that names concrete implementations**. Swapping NIM↔OpenAI or local↔NIM reranker is a one-line config change, and adding a new vector store is a single file plus one registry branch.

**Design principle:** one env var flips a provider; nothing outside `core/config.py` reads env directly.

---

## 2. Status at a glance

| Aspect | State |
|---|---|
| **Architecture** | ✅ Complete — all interfaces frozen, all components implemented |
| **Tests** | ✅ 188 test functions across 11 files, passing |
| **CI** | 📋 Gating plane designed, spec + plan written (SP5) — awaiting implementation |
| **Production-readiness** | 🔴 **NOT production-grade** — see §8. 10 critical + 10 high findings from the audit |
| **Eval baseline** | ✅ Langfuse-native eval (`eval.experiment` + `eval.gate` + `eval.dataset_cli`); a live `baseline` dataset run still needs producing |
| **Semantic cache** | ✅ Decomposition E implemented (plumbing + fake + live smoke) — opt-in, off by default; live baseline / enabling in a real environment deferred |
| **Observability** | ✅ Langfuse v4 (OTel) wired; ⚠️ off by default, leaks raw queries when on |
| **Git** | Private repo `git@github.com:ShreytamGoyal/production-rag.git`; `main` at `0e16528` (2026-07-24). |

**One-line verdict (reconciled 2026-07-24):** A real, working pipeline whose core safety/correctness gaps from the original audit are now **closed in code** — auth, output-block containment, ingest PII redaction, pre-trace query redaction, and true hybrid retrieval all landed (see §8). What remains before real traffic is operational, not safety-critical: a live eval `baseline` (Decomposition D) so the CI gate can catch regressions, a Qdrant-client timeout/retry, and deploy packaging (Dockerfile, rate limiting). _(The detailed line-by-line auto-audit `PRODUCTION_READINESS_AUDIT.md` was a session-time artifact and is not committed to the repo; §8 below is the authoritative, current summary.)_

---

## 3. Tech stack

### Language & tooling
- **Python 3.11–3.13**
- **Package manager:** `uv` (shared venv, `uv sync --all-extras`)
- **Build backend:** hatchling
- **Lint/format:** ruff (line length 100)
- **Tests:** pytest + pytest-asyncio

### Core libraries
| Concern | Library |
|---|---|
| Config / models | `pydantic` v2, `pydantic-settings` |
| LLM + embeddings client | `openai` SDK (pointed at NVIDIA NIM by default), `anthropic` SDK |
| Vector store | `qdrant-client` |
| Document registry | `psycopg[binary]` (Postgres) |
| Sparse retrieval | `rank-bm25` |
| Tokenization | `tiktoken` |
| Numerics / stats | `numpy`, `scipy` (eval), `pandas` (eval) |
| HTTP / retry | `httpx`, `tenacity` |
| Local reranker | `sentence-transformers` (BGE cross-encoder) |
| API / console | `fastapi`, `uvicorn` (console is static HTML served by the API) |
| Observability | `langfuse` (v4, OpenTelemetry-based) |
| Datasets | HuggingFace `datasets` |
| Optional cross-check | `ragas` + `langchain-openai` (native metrics are the spine; ragas is an optional sanity check) |
| Semantic cache backend | `redisvl` (Redis 8; optional `cache` extra, lazily imported) |

### Models (defaults — from `core/config.py`, the source of truth)
| Role | Default (NIM) | Anthropic alt |
|---|---|---|
| Embeddings | `baai/bge-m3` (1024-d, multilingual) | — (or OpenAI `text-embedding-3-large`) |
| Generation | `meta/llama-3.3-70b-instruct` | `claude-sonnet-4-6` |
| Contextual prefix (cheap) | `meta/llama-3.1-8b-instruct` | `claude-haiku-4-5` |
| LLM judge / RAGAS backing | `meta/llama-3.3-70b-instruct` | `claude-sonnet-4-6` |
| Reranker | local `BAAI/bge-reranker-v2-m3` | NIM `nvidia/llama-3.2-nv-rerankqa-1b-v2` |

> ⚠️ **Doc drift:** the README's Stack table lists the embedder as `nv-embedqa-e5-v5`, but the actual config default is `baai/bge-m3`. The README also has a **TBD metrics table**. Treat `core/config.py` as authoritative over the README.

### Swappable-provider matrix (one env var each)
| Concern | Default | Swap to | Switch |
|---|---|---|---|
| Generation | NIM llama-3.3-70b | Anthropic Claude | `GEN_PROVIDER=anthropic` |
| Embeddings | NIM bge-m3 (1024-d) | OpenAI 3-large (3072-d) | `EMBED_BASE_URL` + `EMBED_MODEL` + `EMBED_DIMENSION` |
| Vector store | Qdrant | add a `VectorStore` impl + registry branch | `VECTOR_STORE` |
| Reranker | BGE local | NIM rerank | `RERANKER=nim` |
| Observability | off | Langfuse self-host | `LANGFUSE_ENABLED=true` |

### Infrastructure (`infra/docker-compose.yml`)
- **App backends:** Qdrant (`:6333`, vector store), Postgres (`:5432`, document registry), Redis 8 (`:6379` — arq queue + optional semantic cache)
- **Langfuse v3 self-host stack (6 services):** web + worker + postgres + clickhouse + redis + minio
  (the `redis` here is the **same single Redis 8 container** listed under app backends above — one
  container, both roles: arq queue + semantic cache, and Langfuse's own queue/cache)
- **CI:** GitHub Actions — `eval-gate.yml` (lint + pytest on every PR; eval job intended to gate on regression)

> **Note:** A baked application `Dockerfile` (+ `.dockerignore`) now exists at the repo root, and `infra/docker-compose.yml` has an `api` service that builds it (SP9, Docker-only scope). It runs the FastAPI app via `uvicorn app.api:app`; the same image also runs the `ingest-worker` via a command override. `api` and `ingest-worker` share ingest state through the `app_shared_cache` named volume. Rate limiting / request caps and other hardening are still deferred (see §9 item 4).

---

## 4. Repo layout

```
core/            Interfaces (Protocols), types, config, registry, pipeline, RRF, context assembly
providers/       Concrete impls behind the interfaces:
  embedders/     openai_compatible (NIM/OpenAI)
  generators/    openai_compatible, anthropic
  rerankers/     local_cross_encoder, nim_rerank
  vectorstores/  qdrant_store
  sparse/        bm25
retrieval/       hybrid (dense+sparse→RRF→rerank), acl (filter builder)
generation/      grounded_generator (cited, structured output), prompts
guardrails/      runner + input_injection, pii_guard, output_groundedness,
                 citation_enforcement, schema_validation
ingest/          chunking, contextual (LLM prefix), pii, base, run (CLI)
corpora/         hotpotqa/, arxiv/, financebench/ — one adapter each
eval/            retrieval_metrics, generation_metrics, llm_judge, stats (bootstrap CI),
                 experiment (Langfuse runner), gate, evaluators, langfuse_eval +
                 _langfuse_backend (SDK seam), dataset_cli (seed/add-from-trace),
                 fast_subset, ragas_adapter
observability/   langfuse_tracing (v4 OTel), cost, dashboard
app/             api (FastAPI /query), documents (async ingest), ui (/ui test console)
cache/           semantic_cache (Protocol + serialization + build_cache),
                 _redisvl_backend (only redisvl importer, lazy)
tests/           11 files, 188 test functions, _fakes.py for injection;
                 tests/cache/ (FakeSemanticCache + offline cache suite)
infra/           docker-compose.yml (Redis 8), .env.example
docs/            superpowers/{specs,plans}/ (cache design + plan),
                 architecture.md, PROJECT_STATUS.md (this file)
.github/         workflows/eval-gate.yml
```

### Data flow
**Ingest:** raw docs → chunker → (optional) LLM contextual prefix per chunk → embedder → vector store + BM25 index; golden eval set written to `data/eval/<dataset>.json`.

**Query:** request → ACLContext (from caller) → [input guardrails] → embed query → dense search + BM25 search (both ACL-filtered) → RRF fusion (k=60) → cross-encoder rerank → token-budgeted context assembly → grounded generator (structured output + inline citations) → [output guardrails] → Answer + citations. Every stage traced to Langfuse with token/latency/cost.

---

## 5. How to run it

```bash
make install                              # uv sync --all-extras
cp infra/.env.example .env                # then set NVIDIA_API_KEY=nvapi-...
make up                                    # Qdrant + Postgres + Langfuse stack
make ingest DATASET=hotpotqa               # or: uv run python -m ingest.run --dataset hotpotqa --limit 200
make seed DATASET=hotpotqa ITEMS=data/eval/hotpotqa.json   # upload golden items to Langfuse
make eval DATASET=hotpotqa RUN=baseline                    # run experiment -> Langfuse dataset run
make eval DATASET=hotpotqa RUN=candidate
make gate DATASET=hotpotqa RUN=candidate                   # vs "baseline" run; exits nonzero on regression
make console                               # FastAPI :8000 + test console at /ui
make api                                   # FastAPI :8000
make test                                  # pytest
```

Pass `--contextual` to `ingest.run` to enable per-chunk LLM prefixes (~1 LLM call/chunk — respect NIM's ~40 rpm free-tier cap with `--limit`).

---

## 6. Key configuration knobs (`core/config.py`)

| Knob | Default | Notes |
|---|---|---|
| `VECTOR_STORE` | `qdrant` | Qdrant-only (`Literal["qdrant"]`); add a store via the `VectorStore` Protocol |
| `EMBED_MODEL` / `EMBED_DIMENSION` | `baai/bge-m3` / `1024` | dimension drives store schema — never hardcoded |
| `GEN_PROVIDER` / `GEN_MODEL` | `openai` / `meta/llama-3.3-70b-instruct` | `openai` = OpenAI-compatible (NIM) |
| `RERANKER` | `local` | or `nim` |
| `rrf_k` / `retrieve_top_k` / `rerank_top_n` | `60` / `20` / `8` | metrics reported at k=5 because rerank_top_n=8 |
| `context_token_budget` | `4000` | token-budgeted assembly |
| `request_timeout_seconds` / `max_retries` | `600.0` / `5` | OpenAI SDK backoff (NIM can be slow) |
| `guardrails_enabled` | `True` | forced **off** on the eval path (would confound metrics) |
| `langfuse_enabled` | `False` | |
| `max_chunks_per_corpus` | `2000` | keeps corpora inside NIM rate limits |
| `cache_enabled` | `False` | opt-in semantic cache; needs Redis 8 + `uv sync --extra cache` |
| `cache_similarity_threshold` | `0.9` | min cosine similarity for a cache hit |
| `cache_ttl_seconds` | `3600` | per-entry TTL (new-document staleness backstop) |
| `rewriter_enabled` | `True` | SP12 query rewriter (synonym tier); retrieval-only, generation keeps the original question |
| `rewriter_llm_enabled` | `True` | LLM expansion fallback for ≥ threshold-word queries with no synonym hit |
| `rewriter_llm_threshold` | `5` | word count above which the LLM tier may fire |

Keys: component keys (`EMBED_API_KEY`, etc.) fall back to a shared `NVIDIA_API_KEY` (or `OPENAI_API_KEY` when the base URL points at openai.com). Config is loaded from `infra/.env` then `.env` (root wins).

---

## 7. What's built (feature inventory)

| Capability | Status | Notes |
|---|---|---|
| Swappable Protocol interfaces + registry | ✅ | The core abstraction; works |
| Dense retrieval (Qdrant) | ✅ | ACL filter applied server-side, pre-similarity |
| Sparse retrieval (BM25) | ✅ in eval / 🔴 dead in API | See audit — API builds it empty |
| RRF fusion (k=60) | ✅ | |
| Cross-encoder reranking (local + NIM) | ✅ | |
| Contextual chunking (Anthropic technique) | ✅ | disk-cached; ingest-path robustness gaps |
| Grounded generation + inline citations | ✅ | structured output; citation validity gap (audit) |
| Multi-tenant ACL isolation | ✅ store-side / 🔴 spoofable at API | filters correct; **no auth** derives the tenant |
| Guardrails (injection, PII, groundedness, citation, schema) | ⚠️ wired but partly theatrical | output BLOCK leaks content; injection is regex-only |
| Eval harness (retrieval + RAGAS-style + judge + bootstrap CI) | ✅ | native metrics, now run as Langfuse experiments (client-side scores) |
| CI eval gate | ✅ Langfuse-native | `eval.experiment` → `eval.gate` (paired-bootstrap and/or thresholds, scores read back from Langfuse); needs a `baseline` dataset run |
| Observability (Langfuse v4, cost) | ⚠️ SP7 | works, but off by default and leaks raw queries when on. Cost: `observability/cost.py:PRICING` is now the complete, authoritative hardcoded table for every default model in `core/config.py` (embed/gen/context/judge, Anthropic variants, both rerankers), and `cost_usd()` logs a one-time warning per unknown model so a missing entry is visible instead of silently costing $0. **Still open:** `core/pipeline.py:249` looks up cost by `settings.gen_model` (the NIM identifier) even when `gen_provider="anthropic"` actually served the call, so the Anthropic path can price against the wrong entry — a pipeline wiring bug, not a pricing-table gap |
| Semantic cache (answer + retrieval tiers, Redis 8/redis-vl) | ✅ Decomposition E | plumbing + `FakeSemanticCache` + `RedisVLSemanticCache` + live smoke; tenant/collection-isolated, targeted eviction + TTL backstop, eval bypass; **off by default** — enabling against a live Redis 8 + producing a cache-hit baseline is deferred |
| Query rewriting (per-tenant synonym + LLM expansion) | ✅ SP12 | `HybridQueryRewriter`; runs after input-guard redaction, before the cache-key embed; retrieval-only (generation keeps the original question); hostile per-tenant synonym isolation; fail-soft; on by default; enabled in eval (G4). Synonym dictionaries are read from Redis (`rewriter:synonyms:{tenant_id}`) — no admin load path yet |

> **redis-vl call surface verified (2026-07-24):** the live smoke test `CACHE_LIVE_SMOKE=1 REDIS_URL=redis://localhost:6399 .venv/bin/python -m pytest tests/test_cache_live_smoke.py` passed against a real Redis 8.8.0 (`redis:latest`, query engine in core) with redis-vl 0.23.0 — `index.load(..., ttl=)`, `index.drop_keys(...)`, `index.create(overwrite=False)`, `VectorQuery`/`FilterQuery`/`Tag` field names, and cosine `vector_distance` in `[0, 2]` all confirmed. The gate caught one real bug: the payload text field could not be named `payload` (reserved keyword arg on redis-py's search `Document`, collides at result parse) — renamed to `cache_payload`. The cache remains default-off (`CACHE_ENABLED=false`); flipping it on + producing a live cache-hit baseline is still deferred. Note: homebrew's `redis@8` is built without the query engine (only the `vectorset` module), so use the official `redis:8`/`redis:latest` image, not a homebrew server.

**Measured numbers (local, not committed):** HotpotQA baseline recall@5 ≈ 0.93, MRR ≈ 0.98, nDCG@5 ≈ 0.90 (N=50, bge-m3). The README metrics table is still TBD and no `baseline` Langfuse dataset run has been produced yet (eval plumbing is in place; running a live baseline is a separate step).

---

## 8. What's NOT production-grade (audit summary)

> **Reconciled 2026-07-24 against current `main` (`0e16528`).** The original
> auto-audit (`PRODUCTION_READINESS_AUDIT.md`, 96 findings — a session-time
> artifact, **not committed to this repo**) predates the A–E decompositions and
> the security/pivot work; **5 of its 7 "P0 blockers" are now fixed in code** and
> are struck through below with the fix location. This section is the current,
> authoritative truth.

### P0 blockers — reconciled status
1. ~~**No authentication — tenant is client-controlled**~~ → ✅ **FIXED.** Tenant identity comes ONLY from a cryptographically verified JWT (`app/auth.py:require_principal`, wired at `app/api.py:76`); `QueryRequest` carries no identity field; missing/invalid token → 401.
2. ~~**Blocked answers returned verbatim**~~ → ✅ **FIXED.** An output-guardrail BLOCK overwrites `ans.text` with a generic refusal and scrubs every metadata copy (`core/pipeline.py:258-274`); the block reason stays only in the trace.
3. ~~**PII never redacted at ingest**~~ → ✅ **FIXED.** `_apply_pii_ingest_policy` redacts document text before chunking and **fails closed** on detector error (`ingest/run.py:38-109`); `pii_mode` defaults to `"redact"` (`core/config.py:137`). `keep` mode tags + audits instead.
4. ~~**Raw query logged to Langfuse pre-redaction**~~ → ✅ **FIXED.** The input guard runs and redaction is applied BEFORE the root trace span is created; a blocked query traces as `"[BLOCKED]"` (`core/pipeline.py:109-137`).
5. ~~**API silently runs dense-only**~~ → ✅ **FIXED.** `app/api.py:33` builds `version="full"` (hybrid); the sparse tier is a per-tenant `TenantSparseStore` resolved at query time (not a build-time `dataset` pickle), so BM25 is populated per caller.
6. **CI eval gate can't catch regressions until a baseline run exists.** ❌ **STILL OPEN** — the one genuine P0 left. The Langfuse-native gate (`eval.gate`) needs a `baseline` dataset run on hosted Langfuse (+ `LANGFUSE_*` / `NVIDIA_API_KEY` repo secrets); until one exists the gate has nothing to compare against. **= Decomposition D.**
7. **Timeouts/retries on hot-path calls.** ⚠️ **MOSTLY FIXED.** Embedder + generator use the OpenAI SDK's `max_retries=5` and `request_timeout_seconds` (`core/config.py:87-88`, `core/registry.py:101-102`); the NIM reranker uses a `tenacity` `@retry` with exponential backoff (`providers/rerankers/nim_rerank.py:31`). **Residual:** the Qdrant client is constructed with no explicit timeout and no retry wrapper (`providers/vectorstores/qdrant_store.py:71`), and the default `request_timeout_seconds=600.0` is very generous for a query hot path.

### Recurring themes — reconciled
- **Guardrails** — output-block leak is fixed (see #2); an **indirect-injection scan** now runs over retrieved chunk text and flags `indirect_injection_suspected` (`core/pipeline.py:184-189`, detection + log, non-blocking). Input injection is still primarily regex-based with optional LLM escalation — hardened, not exhaustive.
- **"Green but wrong"** — dense-only fallback fixed (#5); **NaN-passes-gate fixed** (`eval/gate.py:74-82` treats `nan` CI-bound / thresholds as non-passing); **incomplete cost table fixed (SP7)** — `PRICING` now covers every default model and unknown models warn once instead of silently pricing at $0. Not re-checked this pass: whether tracing over-swallows errors. Still open: `core/pipeline.py:249` prices by `gen_model` regardless of `gen_provider`, so the Anthropic path can look up the wrong PRICING entry (see §7 Observability row).
- **Security-critical paths in CI** — the `acl-isolation` CI job now runs the **real** ACL/isolation suites (`tests/test_stores_acl.py`, `tests/test_multitenant_isolation.py`) against live Qdrant + Postgres service containers (Qdrant readiness gated host-side after the healthcheck fix). Real filters are exercised in CI, not just a fake.
- **Not packaged for deploy** — config **is** validated at boot (4 `@model_validator(mode="after")` in `core/config.py`) and a `/healthz` exists (`app/api.py:68`). A baked `Dockerfile` + `api` compose service now exist (SP9). **Still open:** no true rate limiting (only an 8000-char question cap, `max_question_chars`).

> ✏️ **Note on the `infra/.env` NVIDIA key:** the auto-audit flagged it "rotate today." It is **gitignored and never tracked in git**, and also excluded from the Docker build context via `.dockerignore` — local hygiene, not an emergency.

---

## 9. Open work / next steps

The P0 security/correctness batches from the original audit are **done** (see §8, items #1–#5). What genuinely remains:

**Priority order (recommendation):**
1. **Make the eval gate real — Decomposition D (#6).** Seed the hotpotqa Langfuse dataset, add `LANGFUSE_*` + `NVIDIA_API_KEY` repo secrets, run `eval.experiment --run-name baseline` on hosted Langfuse; then CI's `eval.gate` has a baseline to compare against. **This is the highest-value remaining task** — it's what makes the whole CI gate functional.
2. **Semantic cache go-live.** `uv sync --extra cache`, flip `CACHE_ENABLED=true` in a real env, and measure a hit-rate/latency baseline. The redis-vl call surface is already verified (live smoke passed, PR #12); rides naturally on D's baseline infra.
3. **Close the #7 residual.** Give the Qdrant client an explicit timeout + a retry wrapper, and reconsider the 600s default timeout for the query hot path.
4. **Deployability (partially done — SP9).** A baked `Dockerfile` (+ `.dockerignore`) and `api` compose service now exist (§3). Still deferred: real rate limiting / request caps, k8s/autoscaling, CI/CD image publishing, reverse proxy/TLS, secrets managers. (Config boot-validation and `/healthz` already exist.)
5. **Deferred perf nit (E).** A cache miss re-embeds the query the retriever also embeds — thread the query vector into the retriever interface to avoid the double embed.

**Parked:** full-suite eval baseline → README metrics table; stand up the 6-container Langfuse stack for a live trace; `full`-version run + `eval.gate` lift table.

---

## 10. Pointers

| What | Where |
|---|---|
| Production readiness (current) | §8 of this file (reconciled 2026-07-24). The line-by-line `PRODUCTION_READINESS_AUDIT.md` was a session-time artifact and is **not committed**. |
| Cache design spec (superseded) | `docs/superpowers/specs/2026-06-27-rag-cache-design.md` |
| Cache implementation plan (superseded, 12 tasks) | `docs/superpowers/plans/2026-06-27-rag-cache.md` |
| Semantic cache design (Decomposition E, implemented) | `docs/superpowers/specs/2026-07-24-decomposition-e-semantic-cache-design.md` |
| Semantic cache implementation plan (Decomposition E, 7 tasks) | `docs/superpowers/plans/2026-07-24-decomposition-e-semantic-cache.md` |
| Semantic cache architecture | `docs/architecture.md` — "Semantic Cache" section |
| Query rewriting design (SP12, implemented) | `docs/superpowers/specs/2026-07-12-sp12-query-rewriting-design.md` |
| Query rewriting plan (SP12, 5 tasks; rewritten 2026-07-24 for current arch) | `docs/superpowers/plans/2026-07-12-sp12-query-rewriting.md` |
| Query rewriting architecture | `docs/architecture.md` — "Query Rewriting" section |
| Original build plan | (plan-mode file, referenced in session history) |
| Commit rules | `CLAUDE.md` — commits authored solely as Shreytam Goyal; no Claude attribution |
| Config source of truth | `core/config.py` |
| Provider wiring | `core/registry.py` |
| Query pipeline | `core/pipeline.py` |
