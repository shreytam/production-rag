# Production RAG — Async PDF Ingestion · Hybrid Retrieval · Semantic Cache

A production-grade Retrieval-Augmented Generation service built behind clean,
swappable interfaces (no framework lock-in in the core). It pairs a real
retrieval pipeline — async **PDF document** ingestion, hybrid search, reranking,
multi-tenant isolation — with production discipline: JWT auth, server-side ACL
isolation, guardrails, and opt-in observability.

## Overview

Most RAG demos glue an LLM to a vector database and stop there. This system
adds the engineering discipline required before production:

- **Async document ingestion API** — authenticated tenants `POST /documents`;
  the request returns `202` immediately and an `arq`/Redis worker parses,
  chunks, embeds, and indexes in the background. Incremental re-ingest and
  async delete are content-hash aware.
- **Hybrid retrieval** (dense + BM25 + RRF) with cross-encoder reranking.
- **Contextual chunking**: an optional LLM-generated prefix per chunk improves
  recall (following Anthropic's contextual-retrieval technique).
- **Server-side ACL isolation**: every chunk carries a `tenant_id`; the filter
  is applied **before** similarity and is derived from a cryptographically
  verified JWT, never from the prompt text.
- **Grounded, cited generation**: structured output enforces inline citations;
  the generator refuses when context is insufficient.
- **Guardrails**: prompt-injection heuristics + indirect-injection scanning on
  input; PII redaction at ingest (fail-closed); output-groundedness and
  citation enforcement on the way out.
- **Two-tier semantic cache** (Redis 8 / redis-vl): answer-level and
  retrieval-level, tenant/collection-isolated, with targeted eviction on
  document change and a TTL backstop. Opt-in, off by default.
- **Native metrics library** (`generation/metrics.py`) — e.g. faithfulness,
  implemented natively with no external dependency, fully injectable for
  offline testing.
- **PDF-only documents**: tenants upload PDFs via `POST /documents` (multipart)
  or bulk-ingest a directory with `python -m ingest.run --input`. The benchmark
  corpus adapters and the Langfuse-native eval harness were removed in the
  pivot to PDF documents.

---

## Architecture

### Mermaid diagram (renders on GitHub)

```mermaid
flowchart TD
    subgraph Ingest["Async Ingestion (arq worker)"]
        UP[POST /documents\nJWT-authenticated] --> BLOB[Blob store\nlocal disk]
        UP --> Q[(Redis queue\narq)]
        Q --> W[Ingest worker]
        W --> P[Parser\nplain-text / unstructured]
        P --> CH[Chunker]
        CH --> CX[Optional: Contextual Prefix\nLLM per chunk]
        CX --> EM[Embedder\nNIM baai/bge-m3 · 1024-d]
        CH --> BM[Per-tenant BM25 index]
        EM --> VS[(Vector Store\nQdrant)]
        W --> REG[(Doc registry\nPostgres)]
    end

    subgraph Query["POST /query"]
        G[User Query] --> AUTH[JWT principal\n→ ACL context]
        AUTH --> CACHE{Semantic cache?\nRedis 8 / redis-vl}
        CACHE -->|hit| ANS[Answer + Citations]
        CACHE -->|miss| I[Dense Retriever\nEmbedder + Qdrant]
        AUTH --> J[Sparse Retriever\nper-tenant BM25]
        I --> K[RRF Fusion\nk=60]
        J --> K
        K --> L[Cross-Encoder Reranker\nBGE-local or NIM]
        L --> M[Context Assembly\ntoken-budgeted]
        M --> N[GroundedGenerator\nStructured output + citations]
        N --> ANS
    end

    N --> OBS[Langfuse\ntraces · cost · latency]
```

### ASCII diagram (quick reference)

```
POST /documents (202) → blob store + arq queue
        └─ worker: parse → chunk (+contextual prefix) → embed → Qdrant (+per-tenant BM25)

POST /query → JWT → ACL context → semantic cache?
      ├─ hit  → cached Answer
      └─ miss → hybrid retrieve (dense + BM25) → RRF fuse → rerank (cross-encoder)
                → ACL filter (pre-similarity, server-side) → context assembly (budgeted)
                → generate (grounded, cited, structured output) → Answer + citations
                                                 │
         traces + cost + latency metrics → Langfuse dashboard
```

---

## Repo Layout

```
.
├── core/                 # Contracts, types, config, registry, pipeline
│   ├── interfaces.py     # Protocol definitions (Embedder, VectorStore, SparseRetriever, …)
│   ├── types.py          # Shared data models (Chunk, Answer, ACLContext, …)
│   ├── config.py         # All env knobs via pydantic-settings (Settings) — source of truth
│   ├── registry.py       # Single place that names concrete implementations
│   ├── pipeline.py       # RAGPipeline: cache-aware answer(), hybrid retrieve + rerank
│   ├── rrf.py            # Reciprocal Rank Fusion
│   └── context_assembly.py
│
├── providers/            # Swappable implementations behind the Protocols
│   ├── auth/             # jwt_verifier, dev_signer, allowlist
│   ├── embedders/        # openai_compatible (NIM / real OpenAI)
│   ├── generators/       # openai_compatible.py, anthropic.py
│   ├── rerankers/        # local_cross_encoder (BGE), nim_rerank
│   ├── sparse/           # bm25, per-tenant store, pickle loader
│   ├── vectorstores/     # qdrant_store.py
│   ├── docstore/         # postgres, memory (document registry)
│   ├── manifest/         # jsonl_store (chunk manifest per document)
│   ├── blobstore/        # local_disk (uploaded file bytes)
│   └── pii/              # regex_detector, presidio_detector (optional)
│
├── ingest/               # Ingestion pipeline
│   ├── worker.py         # arq worker: ingest_document / delete_document tasks
│   ├── incremental.py    # Content-hash-aware (re)ingest + delete
│   ├── run.py            # CLI: python -m ingest.run --input <dir-of-pdfs>
│   ├── parsers/          # plain_text, unstructured
│   ├── chunking.py       # Fixed-size + overlap chunker
│   ├── contextual.py     # LLM-generated contextual prefix (cached)
│   ├── pii.py            # PII redaction at ingest time (fail-closed)
│   └── audit.py          # Redaction audit trail
│
├── retrieval/
│   └── hybrid.py         # DenseRetriever, HybridRetriever
│
├── generation/
│   ├── grounded_generator.py   # Token-budgeted assembly + citation enforcement
│   ├── metrics.py              # Native metrics (faithfulness) — survived the eval removal
│   └── prompts.py
│
├── guardrails/
│   ├── runner.py                 # Orchestrates the guardrail stages
│   ├── input_injection.py        # Heuristic + optional LLM injection/jailbreak detection
│   ├── pii_guard.py              # Output PII guard
│   ├── output_groundedness.py    # Answer-vs-context groundedness
│   ├── citation_enforcement.py   # Reconciles [n] markers against context
│   └── schema_validation.py
│
├── cache/                # Semantic cache (opt-in)
│   ├── semantic_cache.py         # Protocol + serialization + build_cache + FakeSemanticCache
│   └── _redisvl_backend.py       # Only redisvl importer (lazy)
│
├── observability/        # langfuse_tracing (v4 OTel), cost, dashboard
├── app/
│   ├── api.py            # FastAPI: /query + /healthz (make api)
│   ├── documents.py      # /documents router (async upload / list / get / delete)
│   ├── auth.py           # require_principal — JWT → tenant identity
│   ├── ui.py             # /ui test console + dev-token endpoint (make console)
│   └── static/
│       └── console.html  # single-page console: upload → ingest → query
│
├── tests/                # Offline test suite (fakes, no secrets, no services)
├── infra/
│   ├── docker-compose.yml  # Qdrant + Postgres + Redis 8 + Langfuse v3 stack
│   └── .env.example
├── .github/workflows/    # CI workflows (lint + tests)
├── Makefile
└── pyproject.toml
```

---

## Stack

| Concern | Default | Swappable to | Switch |
|---|---|---|---|
| Generation | NVIDIA NIM · `meta/llama-3.3-70b-instruct` | Anthropic Claude (Sonnet) | `GEN_PROVIDER=anthropic` |
| Embeddings | NIM · `baai/bge-m3` (1024-d) | OpenAI · `text-embedding-3-large` (3072-d) | `EMBED_BASE_URL` + `EMBED_MODEL` + `EMBED_DIMENSION` |
| Vector store | Qdrant | (add a `VectorStore` impl + registry branch) | `VECTOR_STORE` |
| Reranker | BGE cross-encoder (local, `bge-reranker-v2-m3`) | NIM rerank (`nv-rerankqa-1b-v2`) | `RERANKER=nim` |
| Document registry | Postgres | in-memory (tests) | provider wiring |
| Semantic cache | off | Redis 8 / redis-vl | `CACHE_ENABLED=true` (`--extra cache`) |
| Observability | off | Langfuse (self-host or Cloud) | `LANGFUSE_ENABLED=true` |

All switches are env vars read by `core/config.py` (pydantic-settings).
`core/registry.py` is the only module where concrete class names appear —
nothing outside `core/config.py` reads env directly. No code changes are
needed to flip providers.

> **Note:** the vector store is Qdrant-only in the shipped code (`pgvector`
> was removed). The `VectorStore` Protocol keeps adding another backend to a
> single new file plus one registry branch.

---

## The API

The service is **product-driven**: tenants push their own documents in and query
them, rather than a fixed corpus being pre-ingested. Every route derives its
tenant identity from a verified JWT (`app/auth.py`), never from the request body.

| Method | Route | Behaviour |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (`multipart/form-data`, `application/pdf`). Returns **`202 Accepted`** and enqueues async ingest (parse → chunk → embed → index) on the `arq`/Redis worker. |
| `GET` | `/documents` | List the caller's documents and their ingest status. |
| `GET` | `/documents/{id}` | Fetch one document's metadata / status. |
| `DELETE` | `/documents/{id}` | Returns **`202`** and enqueues async delete (drops chunks from Qdrant + BM25 + registry). |
| `POST` | `/query` | Grounded, cited RAG answer over the caller's own documents. |
| `GET` | `/healthz` | Liveness probe. |

Ingest, re-ingest, and delete are content-hash aware (`ingest/incremental.py`),
so re-uploading an unchanged document is a no-op and edits re-index only what
changed.

---

## Documents

The system is **PDF-document focused** — there are no bundled benchmark corpora
or golden datasets anymore:

- **Per-tenant upload:** authenticated tenants `POST /documents`
  (`multipart/form-data`, `application/pdf`); the `arq` worker parses, chunks,
  embeds, and indexes asynchronously.
- **Bulk ingest:** point the CLI at a directory of PDFs:

  ```bash
  uv run python -m ingest.run --input ./pdfs   # or: make ingest INPUT=./pdfs
  ```

> **Eval harness removed:** the Langfuse-native evaluation stack (dataset CLI,
> experiment runner, retrieval metrics, LLM judge, paired-bootstrap CI gate) was
> removed in the pivot to PDF documents. The native metrics library lives on in
> `generation/metrics.py`.

---

## Quickstart

```bash
# 1. Install (shared uv venv, all extras)
make install                        # uv sync --all-extras

# 2. Configure
cp infra/.env.example .env
# Edit .env — at minimum set NVIDIA_API_KEY=nvapi-...

# 3. Start backends
make up                             # Qdrant + Postgres + Redis + Langfuse stack
# or just the app backends:
make up-app                         # Qdrant + Postgres

# 4a. Product path — run the API and push documents
make api                            # FastAPI on :8000 (uvicorn --reload)
arq ingest.worker.WorkerSettings    # (separate shell) start the ingest worker
#   POST /documents to upload, POST /query to ask

# 4b. Bulk path — ingest a directory of PDFs without going through the API
make ingest INPUT=./pdfs                             # python -m ingest.run --input $(INPUT)

# 5. Test console — upload a document, watch it ingest, query it
make console                        # API + console on http://127.0.0.1:8000/ui
```

The console drives the real HTTP API: it mints a JWT for a tenant you pick, uploads
through `POST /documents`, polls status until the worker reports `ready`, then queries
via `POST /query`. Because every action is an ordinary authenticated request, it
exercises auth, tenant scoping, async ingest, and retrieval rather than an in-process
shortcut. It needs the full stack up (`docker compose -f infra/docker-compose.yml up -d`)
for upload to work end-to-end; `/query` alone only needs Qdrant.

> **Security:** `/ui` and `/ui/token` mint dev credentials, so both return **404**
> unless `AUTH_DEV_SIGNER_ENABLED=true` *and* `JWT_SECRET` are set. `core/config.py`
> refuses to boot with that flag when `APP_ENV=prod`, so a production deploy exposes
> neither route.

> **Note on contextual prefixing:** contextual prefixing fires ~1 LLM call per
> chunk. On large directories, respect NIM's ~40 rpm free-tier cap — see
> `uv run python -m ingest.run --help` for the current flags.

---

## Semantic Cache

An opt-in, two-tier semantic cache (`cache/`) short-circuits repeated work:

- **Answer tier** — a near-duplicate query (cosine ≥ `CACHE_SIMILARITY_THRESHOLD`,
  default `0.9`) returns the cached `Answer` without touching the pipeline.
- **Retrieval tier** — caches the fused/reranked context so generation still runs
  on fresh phrasing while retrieval is reused.

Entries are **tenant- and collection-isolated**, evicted precisely when a
document they depend on changes or is deleted, and expire under a per-entry TTL
(`CACHE_TTL_SECONDS`, default `3600`) as a staleness backstop. Backed by Redis 8
via `redis-vl` (query engine in core); the redis-vl import is fully lazy, so the
offline test suite and lint run without it.

Enable with `uv sync --extra cache` and `CACHE_ENABLED=true` against a real
Redis 8 (the official `redis:8` image — Homebrew's `redis@8` omits the query
engine).

---

## Guardrails & Security

**JWT-derived tenant identity** (`app/auth.py`)
Every request's tenant/ACL context comes only from a cryptographically verified
JWT. `QueryRequest` carries no identity field; a missing/invalid token is a 401.
The tenant is never taken from prompt text or request body.

**Server-side ACL pre-similarity filter**
Every chunk carries `tenant_id` and optional `acl_tags`. The vector store and
BM25 retriever apply ACL as a filter condition **before** scoring, not as a
post-hoc exclusion.

**Prompt-injection / jailbreak detection** (`guardrails/input_injection.py`)
`InjectionGuardrail` scans user input against heuristic patterns
(ignore-previous-instructions, DAN mode, role-override, etc.), with an optional
LLM second-opinion path for ambiguous inputs. A match blocks the request before
it reaches the pipeline.

**Indirect-injection scanning**
Retrieved chunk text is treated as **untrusted data**. A scan over retrieved
content flags `indirect_injection_suspected`, and the generation prompt separates
system instructions from retrieved passages using numbered `[n]` markers, so
payloads embedded in documents are less likely to be read as instructions.

**PII redaction at ingest (fail-closed)** (`ingest/pii.py`)
Document text is redacted before chunking (`pii_mode` defaults to `redact`); a
detector error **fails closed** rather than passing raw text through. Regex
detectors ship by default; Presidio NER is an optional extra. Each redaction is
logged to an audit trail.

**Output guardrails**
An output-guardrail BLOCK overwrites the answer with a generic refusal and
scrubs every metadata copy — the block reason stays only in the trace.
Groundedness and citation-enforcement stages reconcile inline `[n]` markers
against the actual context window; if context is insufficient the generator sets
`refused=True`.

---

## Metrics

The Langfuse-native evaluation harness — dataset CLI, experiment runner,
retrieval metrics, LLM judge, paired-bootstrap confidence intervals, and the CI
gate — was **removed** in the pivot to PDF-only documents. What survives is the
native metrics library in `generation/metrics.py`, including the faithfulness
metric, kept dependency-light and injectable for offline testing.

---

## Observability

Langfuse tracing is opt-in (`LANGFUSE_ENABLED=true`, Langfuse v4 / OpenTelemetry).
When enabled, `observability/langfuse_tracing.py` emits a trace per query
containing:

- Retrieval scores and chunk IDs for each stage (dense, BM25, fused, reranked).
- LLM token counts and estimated cost (`observability/cost.py`).
- End-to-end latency broken down by component.

Queries are redacted **before** the root span is created, and a blocked query
traces as `[BLOCKED]`. The self-hosted Langfuse stack starts with `make up`
(web on port 3000).

---

## What I'd Do at 100× Scale

To scale the RAG pipeline to 100× data volume and traffic, we leverage five operational lessons from Notion's search infrastructure engineering. Because the system is structured around decorator/factory registries and `typing.Protocol` interfaces, scaling is accomplished by writing new backends (providers) and swapping them via environment variables, avoiding edits to the query pipeline (`core/pipeline.py`).

| Operation | Current Limitation | 100× Scaling Architecture | Interface & Seam |
|---|---|---|---|
| **Multi-Tenancy Isolation** | Logical boundaries; all tenant vectors share one collection separated by query filters. | **Physical namespace/collection-per-tenant isolation**. Prevents cross-tenant vector scanning, containerizes data, and contains leak blast-radius. | `VectorStore` Protocol (`core/interfaces.py`). Switched configuration: `tenant_isolation` (`pooled` vs `physical`). |
| **Vector Store Hosting** | Always-on, pay-per-uptime VM instances. | **Serverless Vector DB** backing storage to object storage (e.g. Turbopuffer). Yields ~60% cost reduction by billing for active use instead of idle compute. | `VectorStore` Protocol (`core/interfaces.py`). Registry branch: `providers/vectorstores/turbopuffer_store.py`. |
| **Ingestion Pipeline** | Single async `arq` worker; no streaming or Spark/Ray fan-out. | **Decoupled execution lanes**: Parallel Spark/Ray batch backfills + Kafka CDC streaming lane for low-latency ingest, yielding sub-minute data freshness. | `SparseRetriever` / `VectorStore` write APIs (`ingest/worker.py`). |
| **Embedding Generation** | Paid, rate-limited public APIs; high token costs and latency overhead. | **Self-hosted embedding models** (e.g. Triton Inference Server / vLLM on local GPU instances). Yields ~90% cost reduction and removes call limits. | `Embedder` Protocol (`core/interfaces.py`). Registry branch: `providers/embedders/selfhosted.py`. |
| **Capacity Scaling** | In-place collection re-sharding and online schema/collection migrations. | **Generation-based routing** (monotonic routing by `corpus_version` / `generation`). Point raw writes to a new collection/version index and retire old indices when depleted, avoiding online re-shards. | `VectorStore` / `RAGPipeline` boundaries (`ensure_collection(dimension, version)`). |

Other operational dimensions scale similarly:
- **BM25 retrieval** migrates from in-process per-tenant indexes to a distributed Elasticsearch/OpenSearch cluster using BM25F with tenant-routing.
- **Reranking** runs batched GPU inference via NVIDIA NIM containers to maximize throughput.
- **Semantic cache** promotes the single Redis 8 node to a Redis Cluster with per-tenant keyspaces, already the shape the cache uses today.
- **Rate limiting** moves from the per-instance question cap to a distributed Redis token-bucket middleware at the API boundary.

---

## Make Targets

| Target | Description |
|---|---|
| `make install` | `uv sync --all-extras` — create venv and install everything |
| `make up` | Start Qdrant + Postgres + Redis + the full Langfuse v3 stack |
| `make up-app` | Start app backends only (Qdrant + Postgres) |
| `make up-langfuse` | Start the Langfuse v3 stack (web + worker + db + clickhouse + redis + minio) |
| `make down` | Stop all backend services |
| `make ingest INPUT=<dir>` | Bulk-ingest the PDF documents in a directory |
| `make console` | Launch the API + test console on http://127.0.0.1:8000/ui |
| `make api` | Launch the FastAPI service on port 8000 (uvicorn --reload) |
| `make test` | Run the test suite (`pytest`) |
| `make lint` | Lint with ruff |
| `make fmt` | Format with ruff |
| `make clean` | Remove `.pytest_cache`, `.ruff_cache` |

For more ingest options (batch size, contextual prefixing), call the module
directly:

```bash
uv run python -m ingest.run --help
```

---

## Status

The core safety/correctness work is landed: JWT auth, output-block containment,
fail-closed ingest PII redaction, pre-trace query redaction, and true per-tenant
hybrid retrieval. What remains is operational rather than safety-critical — a
Qdrant-client timeout/retry, semantic-cache go-live, and deploy packaging
(Dockerfile, rate limiting). The benchmark corpora and the Langfuse eval
harness/gate were removed in the 2026-08 pivot to PDF documents. See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the
reconciled, line-referenced breakdown.
