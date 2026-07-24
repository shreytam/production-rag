# Decomposition E — Redis Semantic Cache (Design)

**Status:** Approved (brainstorm complete)
**Date:** 2026-07-24
**Builds on:** `main` @ `4f94bba` (Decomposition D merged)

## 1. Goal

Cut query latency and LLM/retrieval cost by serving **semantically-equivalent
repeat queries** from Redis — without ever serving a **stale** result (citing a
chunk that changed or was deleted) or a **cross-tenant** result.

"Semantic" means a cache hit is decided by **embedding similarity**, not exact
string match: *"how do I reset my password"* and *"I need to change my password"*
resolve to the same cached entry when their query vectors are within a
configurable cosine threshold.

## 2. Scope

**In scope (two cache tiers, semantically matched):**

- **Answer tier** — `query → final Answer`. On a hit, skip retrieval **and**
  generation.
- **Retrieval tier** — `query → reranked chunk set`. On a hit, skip retrieval but
  always regenerate fresh.

**Explicitly out of scope:**

- **Chunk-text cache** (`chunk_id → text`). Redundant here: retrieval already
  returns chunk text inline from the Qdrant payload (`ScoredChunk.chunk.text` is
  populated by the time `retrieve()` returns — the pipeline reads it directly for
  injection scanning and context assembly). There is no separate slow
  hydration step to accelerate. YAGNI unless chunk text later moves to a cold
  store.

**Implementation scope:** *plumbing + fake, one live smoke.* All logic is
unit-tested against an in-memory `FakeSemanticCache` (deterministic offline
suite, no Redis required). The real `redis-vl` backend is exercised by a single
opt-in live smoke test (skipped without Redis 8). Mirrors Decomposition D.

## 3. Key decisions (from brainstorm)

| Decision | Choice | Rationale |
|---|---|---|
| Layers | Answer + Retrieval (skip chunk cache) | Biggest wins; chunk text already inline |
| Match rule | Semantic (embedding similarity) on both tiers | The point of the feature |
| Store | `redis-vl` vector index on **Redis 8** | Purpose-built ANN + TAG filters + TTL; Redis 8 folds the query/vector engine into core, so **no separate Redis Stack image** |
| Invalidation | **Per-document targeted eviction + TTL backstop** | Precise eviction guarantees no dangling citations; TTL covers the new-document blind spot |
| Isolation | `tenant_id` + `collection_id` as mandatory TAG filters | A hit can never cross tenants/collections |
| Default | `cache_enabled = False` (opt-in) | Conservative; requires Redis 8 infra |

## 4. Architecture

New `cache/` subsystem, peer of `eval/` and `retrieval/`, following the project's
swappable-interface pattern.

### 4.1 One interface, two instances, one backend

```
core/pipeline.py ──uses──▶ SemanticCache (Protocol)
                              ├─ answer tier   (index: rag_cache_answer)
                              └─ retrieval tier (index: rag_cache_retrieval)
                                     │
                        ┌────────────┴─────────────┐
             RedisVLSemanticCache            FakeSemanticCache
             (cache/_redisvl_backend.py)     (tests/cache/fake_cache.py)
             ONLY module importing redis-vl  in-memory brute-force cosine
             (lazy, inside methods)          + real reverse index
```

The **same** `SemanticCache` implementation is instantiated **twice** — one per
tier — each bound to its own redis-vl index name. Independent stores, shared
code; that keeps "both, layered" from becoming one tangled unit.

### 4.2 `SemanticCache` Protocol (`cache/semantic_cache.py`)

Generic over a single tier; payload is opaque JSON so the same interface serves
both the answer tier (serialized `Answer`) and the retrieval tier (serialized
`list[ScoredChunk]`).

```python
class SemanticCache(Protocol):
    def lookup(self, *, tenant_id: str, collection_id: str | None,
               embedding: Sequence[float]) -> dict | None:
        """Return the stored payload of the nearest entry within threshold for
        this tenant+collection, else None."""

    def store(self, *, tenant_id: str, collection_id: str | None,
              embedding: Sequence[float], payload: dict,
              doc_ids: Sequence[str]) -> None:
        """Insert an entry tagged with tenant/collection/doc_ids, with TTL."""

    def invalidate_document(self, *, tenant_id: str, collection_id: str | None,
                            doc_id: str) -> int:
        """Delete every entry (this tenant/collection) whose doc_ids TAG contains
        doc_id. Returns count evicted."""
```

Normalized types (`cache/semantic_cache.py`): thin dataclasses/helpers for
(de)serializing `Answer` ↔ payload and `list[ScoredChunk]` ↔ payload via Pydantic
`model_dump`/`model_validate`. `build_cache(settings) -> (answer_cache,
retrieval_cache) | None` lazily imports the redis backend and returns `None` when
`cache_enabled` is False (pipeline then runs cache-free).

### 4.3 Redis backend (`cache/_redisvl_backend.py`)

- **The only module importing `redis-vl`**, imported lazily inside method bodies
  (the same isolation invariant the Langfuse backend follows in `eval/`):
  importing any `cache/` module — and thus lint + the offline suite — needs
  neither Redis nor the `redis-vl` package.
- One `SearchIndex` per tier with schema fields: `vector` (FLOAT32, dim =
  `settings.embed_dimension`, COSINE), `tenant_id` (TAG), `collection_id` (TAG),
  `doc_ids` (TAG, multi-valued), `payload` (TEXT/JSON string). Index created
  idempotently on first use.
- **RediSearch is the reverse index.** `invalidate_document` is a filtered delete
  over `@tenant_id:{..} @collection_id:{..} @doc_ids:{doc_id}` — no separate
  bookkeeping structure.
- **TTL** set per key on `store` (`cache_ttl_seconds`) as the new-document
  backstop.
- Connection reuses `settings.redis_url` / `settings.redis_password`.

### 4.4 `FakeSemanticCache` (`tests/cache/fake_cache.py`)

In-memory: list of entries per `(tenant, collection)`; `lookup` does brute-force
cosine and applies the threshold; `store` appends with a recorded `doc_ids` set;
`invalidate_document` filters the list. TTL is modeled by an injectable clock so
expiry is testable deterministically. No Redis, no `redis-vl`.

## 5. Data flow (`RAGPipeline.answer`, only when a cache is wired)

`build()` passes the already-constructed `embedder` and the two cache tiers to
`RAGPipeline`. The cache is wired **only** when `cache_enabled` is True and this
is not an eval build.

1. **Input guardrails run first** (unchanged); the question is redacted. The cache
   key is computed from the **redacted** question, keeping keys consistent with
   what actually gets retrieved.
2. **Embed the redacted query once** via `self.embedder.embed_query(question)`.
   This single vector serves both tiers and (on a miss) is the same query the
   retriever would compute.
3. **Answer-tier lookup.** Hit → deserialize and return the stored `Answer`; skip
   retrieval, generation, **and output guardrails** (the answer was fully vetted
   when stored — see §7). Trace tag `cache=answer_hit`.
4. Miss → **Retrieval-tier lookup.** Hit → deserialize `list[ScoredChunk]`, skip
   retrieval, proceed to generation. Miss → run retrieval as today, then
   `retrieval_cache.store(embedding, scored_chunks, doc_ids)`. Trace tag
   `cache=retrieval_hit | miss`.
5. Generate; run output guardrails (citation/schema/groundedness) as today.
6. **Store the answer only if not refused and not blocked.** Refusals and
   guardrail blocks are never cached. `answer_cache.store(embedding,
   serialized_answer, doc_ids)`.
7. Return.

`doc_ids` for both tiers = the deduped `doc_id`s of the retrieved/cited chunks
(already computed as `retrieved_doc_ids` in the pipeline).

**Eval bypass:** the eval entry points (`eval/experiment.py` via `pipeline.build`)
construct the pipeline with the cache **off**, exactly as guardrails are forced
off — so cache hits can never confound Langfuse metrics.

## 6. Invalidation & worker integration

- **Ingest worker** (`ingest/worker.py::run_ingest`) and **delete worker**
  (`run_delete`), after they commit changes to Qdrant, call
  `invalidate_document(tenant_id, collection_id, doc_id)` on **both** tiers.
- This precisely evicts every cached answer/retrieval that cites the changed or
  deleted document → **no dangling citations** are ever served.
- **New-document blind spot:** a brand-new doc has no id for the reverse index to
  match, so previously-cached answers that *should now* include it can't be
  targeted. The **TTL** (`cache_ttl_seconds`, default 1 h) bounds that staleness
  window; entries self-heal on expiry.
- Worker builds its own cache instance (gated by `cache_enabled`); it connects to
  the same Redis. When the cache is disabled, the invalidation calls are no-ops.

## 7. Guardrails interaction (deliberate tradeoffs)

- **Answer-tier hits skip output guardrails.** The stored `Answer` passed
  citation/schema/groundedness at store time (only non-refused, non-blocked
  answers are cached). Re-running them on a hit would re-incur the groundedness
  LLM call — defeating the cost win. Documented tradeoff.
- **Input guardrails always run** (before the cache lookup), so injection/PII
  redaction is never bypassed, and the key is computed on the redacted text.

## 8. Configuration (`core/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `cache_enabled` | `False` | Master switch; opt-in (needs Redis 8) |
| `cache_similarity_threshold` | `0.9` | Min cosine similarity for a hit |
| `cache_ttl_seconds` | `3600` | Per-entry TTL (new-doc staleness backstop) |

Reuses existing `redis_url` / `redis_password` and `embed_dimension`.

## 9. Infra & dependencies

- `infra/docker-compose.yml`: bump the app Redis `redis:7` → `redis:8`
  (`redis:8-alpine`). The vector/search engine is in core as of Redis 8 — no
  separate Redis Stack image. arq uses only basic commands and is unaffected.
  Langfuse's bundled Redis is separate and untouched.
- Add `redis-vl` as a **lazily-imported** dependency in its own extra (e.g.
  `cache`) so the base offline suite stays dependency-free. `uv.lock` regenerated.
- `.env.example`: document `CACHE_ENABLED`, `CACHE_SIMILARITY_THRESHOLD`,
  `CACHE_TTL_SECONDS`.

## 10. Testing

**Offline (Fake-backed, deterministic — the CI spine):**

- Threshold: near-duplicate embedding hits; dissimilar embedding misses.
- **Cross-tenant isolation:** tenant B never receives tenant A's cached entry for
  an identical query (security test, matches the project's multi-tenant theme).
- **Cross-collection isolation:** collection-scoped and unscoped queries don't
  share entries.
- Targeted eviction: `invalidate_document` removes exactly the entries citing the
  doc and leaves others intact.
- TTL expiry via injected clock.
- Pipeline behavior: answer-hit skips retrieval **and** generation; retrieval-hit
  skips retrieval only; full miss runs both and populates both tiers.
- Refusals and guardrail-blocked answers are **not** cached.
- Eval build wires **no** cache (bypass).
- Worker calls `invalidate_document` on both ingest and delete after commit.

**Live (opt-in, skipped without Redis 8):** one smoke test doing a real
`redis-vl` store → semantic lookup → `invalidate_document` round-trip against a
Redis 8 container.

## 11. New / changed files

**New**
- `cache/__init__.py`
- `cache/semantic_cache.py` — Protocol, normalized types, serialization,
  `build_cache()`
- `cache/_redisvl_backend.py` — `RedisVLSemanticCache` (only redis-vl importer)
- `tests/cache/__init__.py`, `tests/cache/fake_cache.py`
- `tests/cache/test_*.py` — the offline suite above
- `tests/cache/test_live_smoke.py` — opt-in live test

**Changed**
- `core/config.py` — three cache knobs
- `core/pipeline.py` — cache-aware `answer()`; `build()` wires cache + embedder
- `ingest/worker.py` — `run_ingest`/`run_delete` call `invalidate_document`
- `infra/docker-compose.yml` — `redis:7` → `redis:8`
- `pyproject.toml` / `uv.lock` — `redis-vl` extra
- `.env.example` — cache vars
- `docs/architecture.md`, `docs/PROJECT_STATUS.md` — cache section

## 12. Invariants (to verify at whole-branch review)

1. No top-level `redis-vl` import anywhere in `cache/` — only
   `_redisvl_backend.py`, lazily. Offline suite + lint need neither Redis nor the
   package.
2. A cache hit can never cross `tenant_id` or `collection_id`.
3. No dangling citations: an answer citing a deleted/changed doc is never served
   (targeted eviction), bounded further by TTL for new docs.
4. Refusals / guardrail-blocked answers are never cached.
5. Eval path never uses the cache.
6. Cache disabled by default; the entire subsystem is inert unless
   `cache_enabled` is True.
7. Clean authorship (per CLAUDE.md).
