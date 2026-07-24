# Production RAG — Project Status

_Last updated: 2026-07-11. A cold-start orientation: what this project is, what's built, the tech stack, and what still needs fixing before it's production-grade. Read this first if you've been away from the codebase._

---

## 1. What this project is

A **production-grade Retrieval-Augmented Generation (RAG) system** built behind clean, swappable interfaces (no framework lock-in in the core). The thesis: most RAG demos glue an LLM to a vector DB and stop; this one adds the engineering discipline that comes *before* production — hybrid retrieval, reranking, contextual chunking, multi-tenant ACL isolation, guardrails, and a **rigorous eval harness with bootstrap confidence intervals wired into a CI gate**.

It's a portfolio/foundation project, designed to be the reusable base for two later projects (Graph RAG, Multimodal RAG), so the core pipeline sits behind Python `Protocol` interfaces and **`core/registry.py` is the only module that names concrete implementations**. Swapping Qdrant↔pgvector, NIM↔OpenAI, or local↔NIM reranker is a one-line config change.

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
| **Git** | Clean working tree except untracked `docs/` audit + this file. Private repo `git@github.com:ShreytamGoyal/production-rag.git` |

**One-line verdict:** The system has the *shape* of production RAG and a real, working pipeline — but several advertised safety/quality guarantees are inert or bypassable at the point they matter. It runs; it is not safe to expose. Full detail in [`PRODUCTION_READINESS_AUDIT.md`](./PRODUCTION_READINESS_AUDIT.md).

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
| Vector stores | `qdrant-client`, `psycopg[binary]` + `pgvector` |
| Sparse retrieval | `rank-bm25` |
| Tokenization | `tiktoken` |
| Numerics / stats | `numpy`, `scipy` (eval), `pandas` (eval) |
| HTTP / retry | `httpx`, `tenacity` |
| Local reranker | `sentence-transformers` (BGE cross-encoder) |
| API / demo | `fastapi`, `uvicorn`, `streamlit` |
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
| Vector store | Qdrant | pgvector | `VECTOR_STORE=pgvector` |
| Reranker | BGE local | NIM rerank | `RERANKER=nim` |
| Observability | off | Langfuse self-host | `LANGFUSE_ENABLED=true` |

### Infrastructure (`infra/docker-compose.yml`)
- **App backends:** Qdrant (`:6333`), Postgres/pgvector (`:5432`), Redis 8 (`:6379` — arq queue + optional semantic cache)
- **Langfuse v3 self-host stack (6 services):** web + worker + postgres + clickhouse + redis + minio
  (the `redis` here is the **same single Redis 8 container** listed under app backends above — one
  container, both roles: arq queue + semantic cache, and Langfuse's own queue/cache)
- **CI:** GitHub Actions — `eval-gate.yml` (lint + pytest on every PR; eval job intended to gate on regression)

> **Note:** There is **no Dockerfile for the app itself** yet — the only documented launch is a single-worker `uvicorn` dev server. This is a deployability gap (see audit P1).

---

## 4. Repo layout

```
core/            Interfaces (Protocols), types, config, registry, pipeline, RRF, context assembly
providers/       Concrete impls behind the interfaces:
  embedders/     openai_compatible (NIM/OpenAI)
  generators/    openai_compatible, anthropic
  rerankers/     local_cross_encoder, nim_rerank
  vectorstores/  qdrant_store, pgvector_store
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
app/             api (FastAPI /query), demo (Streamlit)
cache/           semantic_cache (Protocol + serialization + build_cache),
                 _redisvl_backend (only redisvl importer, lazy)
tests/           11 files, 188 test functions, _fakes.py for injection;
                 tests/cache/ (FakeSemanticCache + offline cache suite)
infra/           docker-compose.yml (Redis 8), .env.example
docs/            superpowers/{specs,plans}/ (cache design + plan),
                 PRODUCTION_READINESS_AUDIT.md, PROJECT_STATUS.md (this file)
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
make demo                                  # Streamlit :8501
make api                                   # FastAPI :8000
make test                                  # pytest
```

Pass `--contextual` to `ingest.run` to enable per-chunk LLM prefixes (~1 LLM call/chunk — respect NIM's ~40 rpm free-tier cap with `--limit`).

---

## 6. Key configuration knobs (`core/config.py`)

| Knob | Default | Notes |
|---|---|---|
| `VECTOR_STORE` | `qdrant` | or `pgvector` |
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

Keys: component keys (`EMBED_API_KEY`, etc.) fall back to a shared `NVIDIA_API_KEY` (or `OPENAI_API_KEY` when the base URL points at openai.com). Config is loaded from `infra/.env` then `.env` (root wins).

---

## 7. What's built (feature inventory)

| Capability | Status | Notes |
|---|---|---|
| Swappable Protocol interfaces + registry | ✅ | The core abstraction; works |
| Dense retrieval (Qdrant + pgvector) | ✅ | ACL filter applied server-side, pre-similarity |
| Sparse retrieval (BM25) | ✅ in eval / 🔴 dead in API | See audit — API builds it empty |
| RRF fusion (k=60) | ✅ | |
| Cross-encoder reranking (local + NIM) | ✅ | |
| Contextual chunking (Anthropic technique) | ✅ | disk-cached; ingest-path robustness gaps |
| Grounded generation + inline citations | ✅ | structured output; citation validity gap (audit) |
| Multi-tenant ACL isolation | ✅ store-side / 🔴 spoofable at API | filters correct; **no auth** derives the tenant |
| Guardrails (injection, PII, groundedness, citation, schema) | ⚠️ wired but partly theatrical | output BLOCK leaks content; injection is regex-only |
| Eval harness (retrieval + RAGAS-style + judge + bootstrap CI) | ✅ | native metrics, now run as Langfuse experiments (client-side scores) |
| CI eval gate | ✅ Langfuse-native | `eval.experiment` → `eval.gate` (paired-bootstrap and/or thresholds, scores read back from Langfuse); needs a `baseline` dataset run |
| Observability (Langfuse v4, cost) | ⚠️ | works, but off by default and leaks raw queries when on; $0 cost on Anthropic path |
| Semantic cache (answer + retrieval tiers, Redis 8/redis-vl) | ✅ Decomposition E | plumbing + `FakeSemanticCache` + `RedisVLSemanticCache` + live smoke; tenant/collection-isolated, targeted eviction + TTL backstop, eval bypass; **off by default** — enabling against a live Redis 8 + producing a cache-hit baseline is deferred |

> **Before enabling the semantic cache:** run `CACHE_LIVE_SMOKE=1 .venv/bin/python -m pytest tests/test_cache_live_smoke.py` against a real Redis 8 to verify the redis-vl call surface (`index.load(..., ttl=)`, `index.drop_keys(...)`, `index.create(overwrite=False)`, `VectorQuery`/`FilterQuery`/`Tag` field names, cosine `vector_distance` in `[0, 2]`). The cache stays default-off (`CACHE_ENABLED=false`) until that smoke test has passed.

**Measured numbers (local, not committed):** HotpotQA baseline recall@5 ≈ 0.93, MRR ≈ 0.98, nDCG@5 ≈ 0.90 (N=50, bge-m3). The README metrics table is still TBD and no `baseline` Langfuse dataset run has been produced yet (eval plumbing is in place; running a live baseline is a separate step).

---

## 8. What's NOT production-grade (audit summary)

Full report: [`docs/PRODUCTION_READINESS_AUDIT.md`](./PRODUCTION_READINESS_AUDIT.md) — 96 verified findings (10 critical, 10 high, 60 medium, 16 low).

### P0 blockers (fix before any real traffic)
1. **No authentication — tenant is client-controlled** (`app/api.py:77`). `X-Tenant-Id` header / body sets the tenant; the ACL faithfully scopes to whatever the attacker names → read any tenant's corpus. _Verified._
2. **Blocked answers returned verbatim** (`core/pipeline.py:153`). Output guard sets `refused=True` but never overwrites `ans.text` → the "blocked" content still ships. _Verified._
3. **PII never redacted at ingest** (`ingest/run.py:51`). `PIIRedactor` isn't called; corpus PII embedded + stored in cleartext across all stores.
4. **Raw query logged to Langfuse pre-redaction** (`core/pipeline.py:91`). 100% sample rate; defeats the PII guardrails.
5. **API silently runs dense-only** (`app/api.py:32`). Built with `dataset=None` → BM25 empty → "hybrid" isn't what runs.
6. **CI eval gate can't catch regressions until a baseline run exists.** The Langfuse-native gate (`eval.gate`) needs a `baseline` dataset run on the hosted Langfuse; until one is produced, the gate has nothing to compare against.
7. **No timeouts/retries on hot-path calls** (`retrieval/hybrid.py` + stores/reranker). One slow dependency 500s every query and hangs threads → outage.

### Recurring themes
- **Guardrails are theatrical** — present but bypassable (regex-only injection, no indirect-injection scan, output block leaks content).
- **"Green but wrong"** — dense-only fallback, $0 Anthropic cost, NaN-passes-gate, tracing swallows all errors.
- **Security-critical paths untested in CI** — real ACL filter tests self-skip when no DB is present; the offline isolation test checks a *fake*.
- **Not packaged for deploy** — no Dockerfile, no config validation at boot, no rate limiting / request caps.

> ✏️ **Correction to the auto-audit:** it flagged the `infra/.env` NVIDIA key as "rotate today." Verified: the key is **gitignored and never tracked in git** — it has never left the machine. It's local hygiene (add a `.dockerignore` before containerizing), not an emergency.

---

## 9. Open work / next steps

**Priority order (my recommendation):**
1. **P0 security batch** — auth (#1) + guardrail-leak (#2). Pure security, TDD-able, touches existing code.
2. **P0 correctness batch** — wire BM25 into the API (#5), ingest PII redaction (#3), Langfuse redaction (#4).
3. **Make the eval gate real** — seed the hotpotqa Langfuse dataset, run `eval.experiment --run-name baseline`, then the CI gate (`eval.gate`) has a baseline to compare against (#6). _(Historically blocked on a working NIM key.)_
4. **Resilience** — timeouts + retries + partial-failure fallback on all external calls (#7).
5. **Deployability** — Dockerfile, config validation at boot, real `/healthz`, rate limiting.
6. **Done:** the semantic cache (Decomposition E, `docs/superpowers/plans/2026-07-24-decomposition-e-semantic-cache.md`) — plumbing, `FakeSemanticCache`, `RedisVLSemanticCache`, infra/deps/docs. Remaining: run `uv sync --extra cache` against a real Redis 8, flip `CACHE_ENABLED=true`, and measure a live hit-rate/latency baseline.

**Parked:** full-suite eval baseline → README metrics table; stand up the 6-container Langfuse stack for a live trace; `full`-version run + `eval.gate` lift table.

---

## 10. Pointers

| What | Where |
|---|---|
| Full production audit | `docs/PRODUCTION_READINESS_AUDIT.md` |
| Cache design spec (superseded) | `docs/superpowers/specs/2026-06-27-rag-cache-design.md` |
| Cache implementation plan (superseded, 12 tasks) | `docs/superpowers/plans/2026-06-27-rag-cache.md` |
| Semantic cache design (Decomposition E, implemented) | `docs/superpowers/specs/2026-07-24-decomposition-e-semantic-cache-design.md` |
| Semantic cache implementation plan (Decomposition E, 7 tasks) | `docs/superpowers/plans/2026-07-24-decomposition-e-semantic-cache.md` |
| Semantic cache architecture | `docs/architecture.md` — "Semantic Cache" section |
| Original build plan | (plan-mode file, referenced in session history) |
| Commit rules | `CLAUDE.md` — commits authored solely as Shreytam Goyal; no Claude attribution |
| Config source of truth | `core/config.py` |
| Provider wiring | `core/registry.py` |
| Query pipeline | `core/pipeline.py` |
