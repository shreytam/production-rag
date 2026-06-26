# Production RAG — Multi-Tier Cache Design

**Date:** 2026-06-27
**Status:** Approved design (pre-implementation)
**Author:** Shreytam Goyal

## 1. Problem & Goals

The RAG system currently has **no runtime cache**: identical or near-identical
queries re-run embedding, hybrid retrieval, reranking, and LLM generation every
time. Generation is the dominant cost/latency. We want a production-grade,
multi-tier cache that cuts repeat-query cost and latency without ever serving a
wrong or cross-tenant answer.

**Goals**
- Cut latency and LLM cost for repeated and paraphrased queries.
- Behave like a real production RAG cache: exact + semantic response caching,
  plus embedding and retrieval caches.
- Be **correct and safe by construction**: no cross-tenant leakage, no stale
  answers after re-ingest, no caching of blocked/refused outputs.
- Stay consistent with the project's architecture: one swappable interface,
  config-gated, no-op fallback, traced in Langfuse, off on the eval path.

**Non-goals**
- LRU eviction (the shared Redis must stay `noeviction` for Langfuse — we
  self-bound with TTL + a per-tenant entry cap instead).
- Caching across different models/prompt versions/tenants (those are exact-match
  key dimensions, never merged).
- A separate Redis instance (decision: reuse one shared, latest-version Redis).

## 2. Decisions (locked with user)

1. **Tiers:** multi-tier — L1 exact response, L2 semantic response, L3 query
   embedding, L4 retrieval result.
2. **Backend:** a single **shared Redis, upgraded to the latest major (Redis 8)**,
   serving both Langfuse and this cache. Redis 8 bundles the query engine +
   native vector search into core, so the L2 semantic tier needs no extra image.
3. **Invalidation:** **TTL + version key** — every key embeds `corpus_version`
   and `prompt_version`; a re-ingest or prompt change orphans stale entries
   instantly. TTL is a time-based backstop.
4. **Isolation (hardened from research):** tenant isolation is **hostile by
   design** — `tenant_id` (and model + pipeline + corpus/prompt version) are
   **hashed exactly**; only the query text is compared semantically; the L2 KNN
   is **partitioned per tenant** so a cross-tenant hit is structurally impossible.
5. **Threshold:** default cosine similarity **0.95**, configurable, with an
   offline tuning helper (per research: tune on real traffic, watch the 3–5%
   false-positive ceiling).
6. **Ordering:** input guardrails run **before** the cache; only non-refused,
   non-blocked answers are stored.
7. **Eval-gating:** cache forced **off** on the eval path (a warm cache would
   confound metrics), same pattern as guardrails.

Research backing these (esp. #4–#6): TrueFoundry (text keys are the wrong
default), Portkey & InfoQ (thresholds / false positives), TianPan & Brightlume
(version-based invalidation), arXiv 2601.23088 (semantic key-collision attack).

## 3. Architecture

```
core/interfaces.py        + Cache (Protocol)
providers/cache/
  ├─ redis_cache.py        RedisCache  — Redis 8 client, TTL, vector KNN, ragcache:* keys, DB 1
  └─ null_cache.py         NullCache   — no-op (cache disabled / eval / Redis unreachable)
core/cache.py             CacheKeys (key builder) + SemanticCache (embed→KNN→threshold) + CacheRecord
core/registry.py          build_cache(settings) -> Cache       (redis | null, with graceful fallback)
core/pipeline.py          L1/L2 response check + conditional store, inside tracer spans
providers/embedders/caching.py   CachingEmbedder(Embedder)     # L3 decorator
retrieval/caching.py             CachingRetriever(Retriever)   # L4 decorator
```

- **One Protocol, one real backend, one no-op fallback.** `build_cache` is the
  only place the backend is named (mirrors `build_vector_store`/`build_generator`).
- **Hybrid wiring:** L1/L2 (response) at the pipeline level because a hit skips
  the whole pipeline; L3/L4 as transparent decorators that still satisfy the
  `Embedder`/`Retriever` Protocols.

### 3.1 `Cache` Protocol

```python
@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> dict | None: ...                 # exact GET (L1/L4 records)
    def set(self, key: str, value: dict, ttl: int) -> None: ...  # exact SET with TTL
    def get_vector(self, key: str) -> list[float] | None: ...    # L3 embedding GET
    def set_vector(self, key: str, vec: list[float], ttl: int) -> None: ...
    def knn(self, tenant: str, partition: str, query_vec: list[float],
            top_k: int = 1) -> list[tuple[dict, float]]: ...      # L2: tenant-partitioned, returns (record, cosine_sim)
    def add_semantic(self, tenant: str, partition: str, query_vec: list[float],
                     record: dict, ttl: int, max_entries: int) -> None: ...
    def clear(self, prefix: str = "ragcache:") -> int: ...        # flush (make cache-clear)
```

`NullCache` implements all of these as no-ops (`get*` → `None`, `knn` → `[]`).
`RedisCache` implements them against Redis 8 (string keys + per-key TTL for
L1/L3/L4; a per-tenant vector set / FT index for L2). On init failure
(connection refused, wrong version) it logs once and the registry substitutes
`NullCache` so a query never breaks.

## 4. Keys, records, partitioning

```
version    = "{corpus_ver}.{prompt_ver}"          # bumps invalidate everything stale
partition  = "{version}:{pipeline_ver}:{model}"   # exact-match dimensions (never merged)
exact key  = "ragcache:resp:{partition}:{tenant}:{sha256(norm(question))}"   # L1
emb key    = "ragcache:emb:{embed_model}:{sha256(norm(question))}"           # L3
retr key   = "ragcache:retr:{partition}:{tenant}:{sha256(norm(question))}"   # L4
L2 index   = "ragcache:sem:{partition}:{tenant}"  vector set; payload = CacheRecord
```

- `norm(question)` = trimmed, whitespace-collapsed, lowercased (cheap quality
  control so trivial variants share an entry; per research, low-quality queries
  pollute the cache).
- **CacheRecord** = the serialized `pipeline.run()` dict itself
  (`answer, citations, retrieved_ids, retrieved_chunk_ids, contexts, usage,
  refused`) plus `{created_at, cost_usd, source_query}`. On a hit we return the
  stored `run()` dict verbatim (and a synthesized `Answer` for `answer()`), so a
  cached response is byte-identical to a fresh one.
- **L2 partitioning is the security boundary:** the KNN is issued against the
  `ragcache:sem:{partition}:{tenant}` namespace only. There is no global index
  to filter — cross-tenant or cross-version hits cannot occur. A hit is served
  **only if `cosine_sim ≥ cache_semantic_threshold`**.
- **`corpus_version` is a single global counter**, not per-dataset (the API
  builds with `dataset=None`, so a per-dataset version is ill-defined). Each
  `ingest.run` bumps it and persists it to `.cache/corpus_version` (a Redis
  `ragcache:meta:corpus_ver` mirror is optional); query-time reads it, defaulting
  to `"0"` when the file is absent (fresh checkout / never ingested).
  `prompt_version` is a module constant bumped when `SYSTEM_PROMPT`/prompt
  templates change.

## 5. Request flow (`RAGPipeline.answer`)

```
answer(question, acl):
  with tracer.span("rag.query"):
    # 1. Input guardrails FIRST — a blocked query never touches the cache.
    if guardrails: input check → blocked? return refused (not cached)
    q = norm(question)

    # 2. L1 exact
    rec = cache.get(exact_key(q, acl)) ; if rec: trace(tier=exact); return rec→Answer

    # 3. L3 embedding (also used by L2)
    vec = cache.get_vector(emb_key(q)) or embed_query(q) [→ set_vector]

    # 4. L2 semantic (tenant-partitioned KNN)
    if semantic_enabled:
      hits = cache.knn(acl.tenant, partition, vec, top_k=1)
      if hits and hits[0].sim ≥ threshold: trace(tier=semantic, sim); return hits[0].record→Answer

    # 5. L4 retrieval
    scored = cache.get(retr_key(q, acl)) or retriever.retrieve(...) [→ set]

    # 6. Generate (the expensive call)
    ans = grounded.generate(question, scored)

    # 7. Output guardrails
    if guardrails: output check → may set refused/block_reason

    # 8. Store ONLY clean answers
    if not ans.refused and not blocked:
      cache.set(exact_key, rec, ttl)
      cache.set(retr_key, scored_ids, ttl)
      if semantic_enabled: cache.add_semantic(tenant, partition, vec, rec, ttl, max_entries)

    trace(cache.hit=False, tier=miss)
    return ans
```

Tier-value note (multi-tier was chosen deliberately): L1 is the biggest win; L2
is the production hallmark (highest risk → threshold + partition + margin); L3 is
cheap and feeds L2; **L4 is largely subsumed by L1 for exact repeats** — kept for
completeness and the L2-miss path, and is the cheapest tier to drop later.

## 6. Config knobs

| Knob | Default | Purpose |
|---|---|---|
| `cache_enabled` | `true` | master switch; eval forces off |
| `cache_redis_url` | `redis://:dev-redis-change-me@localhost:6379/1` | shared Redis, logical **DB 1** (Langfuse = 0) |
| `cache_ttl_seconds` | `3600` | TTL backstop (no LRU under `noeviction`) |
| `cache_semantic_enabled` | `true` | L2 toggle (the risky tier) |
| `cache_semantic_threshold` | `0.95` | cosine-sim floor for an L2 hit |
| `cache_semantic_max_entries` | `10000` | per-tenant L2 index cap (self-bound) |

`build(enable_cache: bool | None = None)` — `None` follows `cache_enabled`;
`run_eval.py` and `ragas_adapter.py` pass `False`. `RAGPipeline(cache=None)` (the
direct constructor used by offline tests) means no caching, exactly like
`guardrails=None`.

## 7. Observability

The query's tracer root span gains: `cache.hit` (bool), `cache.tier`
(`exact|semantic|miss`), `cache.similarity` (L2 only), `cache.cost_saved_usd`
(the generation cost a hit avoided). This makes hit-rate and dollar savings
visible in Langfuse alongside the existing latency/cost spans.

## 8. Invalidation & eviction

- **Primary:** version key. Re-ingest bumps the global `corpus_version`
  (`.cache/corpus_version`) → all old `ragcache:*:{old_ver}.*:*` keys are
  orphaned and expire. Prompt change → bump `prompt_version` (constant) → same
  effect.
- **Backstop:** every entry has a TTL (`cache_ttl_seconds`).
- **Eviction:** the shared Redis is `noeviction` (Langfuse requirement), so the
  cache must not depend on LRU. We self-bound via TTL + `cache_semantic_max_entries`
  (oldest semantic entries trimmed when the per-tenant cap is exceeded). Write
  failures (OOM) are caught → that write is skipped, query still succeeds.
- **Manual:** `make cache-clear` → `cache.clear("ragcache:")`.

## 9. Infrastructure changes

- `infra/docker-compose.yml`: bump shared `redis:7` → `redis:8`. Keep
  `--maxmemory-policy noeviction` (Langfuse requires it) and `--requirepass`.
  Redis 8 satisfies Langfuse's "≥ 7" requirement and adds native vector search
  for L2. Set `LANGFUSE_BULLMQ_SKIP_REDIS_VERSION_CHECK=true` on the Langfuse
  services if BullMQ flags the newer version.
- `pyproject.toml`: add `redis>=5` (client) to the `obs`/new `cache` extra;
  `fakeredis>=2` to `dev`.
- `Makefile`: add `cache-clear`.
- `.env.example`: add the six `CACHE_*` knobs.

## 10. Testing strategy

Offline (no services), mirroring the existing fakes pattern:
- **`FakeCache`** (in-memory dict + brute-force cosine) implements the `Cache`
  Protocol → drives pipeline/orchestration tests.
- **Cross-tenant isolation (critical):** tenant A stores an answer; tenant B
  issues the same question → **MISS** on both L1 and L2 (no cross-tenant hit).
  Mirrors `test_multitenant_isolation.py`.
- **Semantic:** paraphrase within threshold → hit; below threshold → miss;
  cross-tenant/cross-version → never hit.
- **Invalidation:** bump `corpus_version` → previous key misses.
- **Don't-cache-bad:** a refused or guardrail-blocked answer is never stored.
- **Warm-cache skips work:** second identical query returns the same answer and
  the generator is **not** called again (assert call count).
- **`RedisCache` unit:** key building + (de)serialization via `fakeredis`.
- **Live Redis-8 vector KNN:** integration test, **skipped** when Redis is
  absent (like the Qdrant/live-NIM tests).

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cross-tenant leak via semantic hit | Per-tenant partitioned KNN (no global index); tenant hashed exactly |
| Wrong answer on look-alike query | Threshold 0.95 + tunable + offline backtest helper; only query text is semantic |
| Stale answer after re-ingest | Version key (corpus+prompt) orphans entries on change; TTL backstop |
| Shared Redis fills (noeviction) | TTL on every entry + per-tenant cap; OOM write caught & skipped |
| Caching a blocked/injection output | Input guardrails before cache; store only non-refused, non-blocked |
| Redis down / wrong version | `RedisCache` init falls back to `NullCache`; queries unaffected |
| Eval metrics confounded by warm cache | Cache forced off on both eval entry points |

## 12. Out of scope / future

- Adaptive/auto-tuned thresholds; per-tenant threshold overrides.
- Caching of intermediate rerank scores.
- A dedicated (non-shared) production Redis with `allkeys-lru`.
- Dropping L4 if telemetry shows it rarely adds hits beyond L1.
