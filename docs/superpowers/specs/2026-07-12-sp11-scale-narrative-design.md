# SP11 · Scale Narrative (VDB-Decision) — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 2 (design artifact — narrative + README section + one decision, **minimal/no code**). This spec **is** the "VDB-Decision" referenced by SP1 (`docs/superpowers/specs/2026-07-12-sp1-security-tenancy-design.md:34,200`) and SP5 (`…sp5-eval-gate-design.md:46`): both defer *physical* namespace-per-org isolation to "VDB-Decision / SP11." It therefore **owns** the physical-isolation decision and **blocks** SP8 (ingest implementation). It touches no security/guardrail/cost/eval behavior — those slices stay authoritative.

---

## 0. What this document is (and is not)

This is a **design/narrative** artifact, not an implementation slice. Its concrete deliverable is (a) this decision doc and (b) a rewrite of the README's "What I'd Do at 100× Scale" section (`README.md:318-331`) from a two-column hand-wave into a defensible architecture. It synthesizes the lessons from Notion's *"Building and scaling Notion's data lake"* / *"Two years of vector search at Notion"* engineering posts and maps them onto **this** system's swappable-Protocol architecture.

It ships essentially **no runtime code**. The `## 10. Files` and `## 9. Testing` sections are deliberately thin (there is little to test in a narrative). The value is in stating, honestly, **what is already designed-for vs. what is aspirational**, and in proving that each 100× move is a *new provider behind an existing Protocol, selected in `core/registry.py`* — not a pipeline rewrite.

> **Why a narrative doc carries file:line rows in §1:** the rows below are not bugs. They are the **current-design limitations at scale** that the 100× architecture addresses, each anchored to the real code that embodies the limitation, so the reader can see exactly which line changes (and, crucially, which lines *do not*).

---

## 1. Context & problem

Line numbers verified against current source at spec time; they may drift as files change.

| # | Scale limitation (not a bug) | Location | Reality today |
|---|------------------------------|----------|---------------|
| 1 | **Tenancy is pooled + logical, not physical.** All tenants share one vector collection; isolation is a *payload filter*, not a boundary. | `providers/vectorstores/qdrant_store.py:69-70` (`self._collection = settings.qdrant_collection`, one collection for all), `core/config.py:39` (`qdrant_collection: str = "rag_chunks"`), `retrieval/acl.py:21-57` (`qdrant_filter` — a `tenant_id` MUST filter) | Every query for every org scans one index and is separated by a `MUST tenant_id == …` predicate. Correct (SP1 hardens it), but a "noisy" or huge tenant's data is co-resident with everyone else's: no per-org index tuning, no per-org cost attribution, no blast-radius containment, and one filter bug = cross-tenant leak. Notion moved to **collection/namespace-per-org physical isolation** for exactly these reasons. |
| 2 | **Vector store is pay-per-uptime, cluster-managed.** | `infra/docker-compose.yml` (Qdrant container, always-on) + `core/registry.py:36-46` (`build_vector_store` → `qdrant`/`pgvector`) | Both current backends (Qdrant, pgvector) are always-running servers you pay to keep hot even when idle, and you scale by provisioning RAM/replicas ahead of demand. Notion reports a **~60% cost cut** (their reported figure) moving to **Turbopuffer** — a serverless, object-storage-backed vector DB that is **pay-per-usage** rather than pay-per-uptime. Our Protocol makes this store swappable; today no serverless provider exists. |
| 3 | **Ingest is sequential and batch-only; no streaming / sub-minute freshness.** | `ingest/run.py:79` (`vectors = embedder.embed_documents(texts)` — one blocking call for the whole corpus), `:87` (`vector_store.upsert(...)` — one blocking upsert), whole module is a one-shot CLI (`:34 main()`) | A document edited now is invisible until someone re-runs `python -m ingest.run` end-to-end. There is no batch/stream split, no incremental upsert, no freshness SLA. Notion runs **decoupled ingest**: a batch lane (Spark/Ray-style backfill) + a streaming lane (Kafka-style CDC) delivering **sub-minute** freshness. |
| 4 | **Embeddings are API-metered, per-token.** | `providers/embedders/openai_compatible.py:1-40` (`OpenAICompatibleEmbedder` → NIM/OpenAI HTTP), `core/config.py:44-48` (`embed_base_url`, `embed_model`) | Every embed is a billed (or rate-limited) network round-trip to NIM/OpenAI; at 100× the embed bill and the ~40 rpm free-tier ceiling dominate ingest wall-clock. Notion reports **~90% cost reduction** (their reported figure) **self-hosting** the embedding model on their own GPUs. Our `Embedder` is a Protocol — a self-hosted embedder is a new provider — but none exists today. |
| 5 | **Capacity is added by re-sharding, not by routing to a new generation.** | `providers/vectorstores/qdrant_store.py:72-93` (`ensure_collection` targets the *single* `self._collection`), `core/interfaces.py:43` (`ensure_collection(self, dimension)` — **no tenant/generation argument**) | To grow the index you re-shard/re-provision the one collection in place — an online migration. Notion's **generation-based routing** onboards new capacity by pointing *new* writes at a fresh generation and letting the old one age out (they cite a **600× onboarding speedup**, their reported figure) — no re-shard. We already have the *seed* of this idea in the **cache's monotonic `corpus_version`** (`docs/superpowers/specs/2026-06-27-rag-cache-design.md` §4/§8): a bump orphans old entries and new writes flow to the new version with no migration. The vector store does not yet apply the same generation discipline. |

**Net:** the system is architecturally *ready* for each of these moves (swappable Protocols + a single registry seam) but has not *taken* any of them. This spec decides the direction and proves the moves are provider-swaps, not rewrites.

---

## 2. Goals

- **Decide** the physical-isolation direction that SP1/SP5 deferred here: adopt **namespace/collection-per-tenant physical isolation** as the target model, behind the existing `VectorStore` Protocol, selected by a new config knob — the current pooled-filter model remaining the **conservative default** until a physical backend ships.
- **Map** the five Notion lessons onto this codebase precisely: for each, name the exact Protocol, the exact `core/registry.py` seam, and the exact lines that stay unchanged.
- **Prove** every 100× move is a *config-level provider swap* — a new class implementing an existing Protocol in `core/interfaces.py`, wired only in `core/registry.py`, with **no change to `core/pipeline.py`** — and be **honest** where that is *not yet* fully true (the `ensure_collection` seam, §5.1).
- **Rewrite** `README.md:318-331` into a credible architecture table that distinguishes *designed-for* from *aspirational*.
- **Draw the two synthesis links** the program cares about: generation-based routing ↔ our cache's `corpus_version`; and Protocol → registry → provider swap.
- Keep it honest: label each capability **designed-for** (Protocol exists, wiring is a swap) or **aspirational** (no provider written yet, and/or a Protocol seam must widen first).

## 3. Non-goals (deferred) — owner named

- **Physical isolation is IN scope for the *decision*** (SP1/SP5 defer it here). What is **OUT**: the *ingest implementation* that would create/populate per-tenant namespaces → **SP8 (Ingest)**. SP11 **blocks** SP8: SP8 must not build multi-collection ingest until this decision is confirmed.
- **The actual streaming/batch ingest pipeline** (Kafka/CDC/Ray workers, incremental upsert, freshness SLA) → **SP8 (Ingest)**. SP11 only states the *target shape* and the Protocol seam it rides on.
- **The runtime cache** (L1–L4, semantic threshold, version invalidation implementation) → the **cache spec** (`2026-06-27-rag-cache-design.md`). SP11 *reuses its `corpus_version` idea* as the routing analogy but changes nothing about it.
- **Tenant-filter correctness / ACL hardening** (fail-closed filters, `MUST tenant_id`, tests) → **SP1 (Security & Tenancy)**. SP11 assumes SP1's pooled-filter guarantees hold and only proposes moving the boundary from *filter* to *physical namespace*.
- **Cost accounting / pricing correctness** of whatever backend is chosen → **SP7 (Observability & Cost)**.
- **Eval-gate behavior** under a new backend → **SP5 (Eval Gate)**. A physical-isolation backend must still pass the SP5 gate; SP11 does not touch the gate.
- **Reranker / generator / BM25 scaling** (GPU batching, OpenSearch BM25F) → future hooks (§11); mentioned in the README rewrite as aspirational, not designed here.

---

## 4. Decisions (PROPOSED) — confirm/override on review

Each row leads with the best-practice option. All are **proposed** for the user to confirm/override.

| # | Decision | Choice (PROPOSED) | Rationale |
|---|----------|-------------------|-----------|
| D1 | Isolation model at scale | **Namespace/collection-per-tenant physical isolation** as the target; keep **pooled-filter as the conservative default** until a physical backend ships. Selected by a new `tenant_isolation` knob. | Physical isolation gives blast-radius containment, per-org index tuning, per-org cost attribution, and makes cross-tenant leak *structurally* impossible (no shared index to mis-filter) — mirroring the cache spec's per-tenant KNN partition. Keeping pooled as default means **no behavior change** on confirm; the switch is opt-in. |
| D2 | Where physical isolation lives | A **new `VectorStore` provider** (e.g. `providers/vectorstores/turbopuffer_store.py`) that routes on `acl.tenant_id` (search) / `chunk.tenant_id` (upsert), wired in `core/registry.py:36`. **No change to `core/pipeline.py` or the retriever.** | The Protocol already threads tenant identity through both hot paths — `search(embedding, top_k, acl)` (`interfaces.py:47`) and `upsert(chunks)` where each `Chunk` carries `tenant_id` (`interfaces.py:45`). A routing store needs **no signature change** on those two. This is the core "config-level, not rewrite" claim. |
| D3 | Serverless / cost backend | Adopt a **serverless, object-storage-backed vector DB (Turbopuffer-class)** as the recommended 100× store, behind the *same* `VectorStore` Protocol; Qdrant/pgvector remain the local/self-host defaults. | Pay-per-usage vs. pay-per-uptime is the ~60% lever Notion reported. Because it's the same Protocol, adopting it is a `build_vector_store` branch + a provider class — `core/pipeline.py` is untouched. **Aspirational:** no such provider is written yet. |
| D4 | Onboarding without re-sharding | **Generation-based routing**: new capacity = a new *generation* namespace that takes new writes; old generation ages out. Reuse the cache's monotonic **`corpus_version`** as the generation counter. | Directly analogous to `2026-06-27-rag-cache-design.md` §4/§8: a version bump orphans stale entries and steers new writes, no migration. Applying the same counter to the vector namespace name (`{collection}:{corpus_version}`) turns re-shard into a pointer flip. **Aspirational** for the store; the counter already exists for the cache. |
| D5 | Ingest decoupling | Target a **batch lane + streaming lane** (Ray/Spark-style backfill + Kafka/CDC-style sub-minute freshness) behind the *same* ingest entry contract; implementation owned by **SP8**. | Matches Notion's decoupled model. The `VectorStore.upsert` Protocol already supports incremental writes (it's an upsert, not a rebuild), so streaming needs no Protocol change — only a caller (SP8) that upserts deltas. **Aspirational.** |
| D6 | Embeddings at scale | Add a **self-hosted `Embedder` provider** (local GPU / Triton / vLLM-served) as a new class behind the `Embedder` Protocol; API embeddings (NIM/OpenAI) remain the default. | ~90% cost cut (Notion's figure) with no interface change: `embed_documents`/`embed_query` are unchanged; only the concrete class differs, selected by `embed_provider`. **Aspirational.** |
| D7 | The one honest seam | **`ensure_collection(self, dimension)` (`interfaces.py:43`) takes no tenant/generation** — so per-tenant physical collections cannot be created through it. Propose **lazy-create-on-upsert** inside the routing store (create the tenant/generation namespace on first write) to keep the Protocol signature stable. | Widening the Protocol signature would ripple to every implementor. Lazy-create confines the change to the *new* provider and preserves the "no interface rewrite" property. This is the single place the "just a config swap" story is not literally free — stated plainly. |
| D8 | Default posture on confirm | Confirming this spec changes **no runtime behavior**: pooled default, API embeddings, batch ingest all stay. It unblocks SP8 and authorizes the future providers above. | A decision doc should not silently flip production. Every new capability is opt-in via a knob defaulting to today's behavior. |

---

## 5. Architecture & components

The architecture is the existing one, unchanged: Protocols in `core/interfaces.py`, concrete impls in `providers/`, wired **only** in `core/registry.py`. This spec proposes *future* providers behind those Protocols; it defines the **seams**, not the implementations.

### 5.1 The tenant-routing / serverless VectorStore (proposed, aspirational)

```
core/interfaces.py     VectorStore Protocol — UNCHANGED for search/upsert/count.
                       (search(embedding, top_k, acl) already carries acl.tenant_id;
                        upsert(chunks) already carries chunk.tenant_id)
                       ⚠ ensure_collection(dimension) has NO tenant arg — see the seam below.
providers/vectorstores/
  turbopuffer_store.py  (PROPOSED, not yet written) TurbopufferStore(settings)
                        - routes to namespace = f"{base}:{corpus_version}:{acl.tenant_id}"
                        - lazy-creates the namespace on first upsert (D7)
                        - serverless, object-storage-backed (D3)
core/registry.py:36    build_vector_store: add a  s.vector_store == "turbopuffer"  branch.
                        THIS IS THE ONLY WIRING CHANGE. pipeline.py is untouched.
```

**The seam, stated honestly (D7):** `VectorStore.search`/`upsert`/`count` all receive tenant identity already (`acl` / `chunk.tenant_id`), so per-tenant *routing* needs no signature change — that half of the claim is fully true. But `ensure_collection(self, dimension)` (`core/interfaces.py:43`) has no tenant/generation parameter, so a physical-per-org store cannot pre-create org collections through the Protocol. Resolution: the routing store **lazy-creates** the `{base}:{version}:{tenant}` namespace on first `upsert`, so the Protocol signature stays stable and the change is confined to the new provider. This is the one place "config-level, not a rewrite" is *almost* free rather than literally free — and it is called out rather than glossed.

### 5.2 Generation-based routing ↔ cache `corpus_version` (the synthesis link)

The cache spec (`2026-06-27-rag-cache-design.md` §4/§8) already runs a **monotonic global `corpus_version`** persisted to `.cache/corpus_version`: each `ingest.run` bumps it, which *orphans* every stale cache key (`ragcache:*:{old_ver}.*`) and steers new writes to the new version — **no migration, no eviction sweep**. Generation-based routing is the same primitive applied to the *vector namespace*: name the collection `{base}:{corpus_version}`, and a version bump means new writes land in a fresh generation while the old one ages out (TTL/retention) instead of being re-sharded in place. One counter, two consumers (cache + store). This is designed-for at the *idea* level (the counter exists) and aspirational at the *store* level (the store doesn't read it yet).

### 5.3 Self-hosted embedder (proposed, aspirational)

```
core/interfaces.py     Embedder Protocol — UNCHANGED (dimension / embed_documents / embed_query).
providers/embedders/
  selfhosted.py         (PROPOSED) a Triton/vLLM-served embedder, same three methods.
core/registry.py:29    build_embedder: branch on a new  embed_provider  knob.
```

No pipeline or ingest change: `ingest/run.py:79` still calls `embedder.embed_documents(texts)` against whatever the registry returns.

### 5.4 Decoupled ingest (proposed, SP8-owned)

The `VectorStore.upsert` contract is already incremental, so a streaming lane is a *new caller* (SP8) that upserts deltas as CDC events arrive — the store, retriever, and pipeline are untouched. The batch lane is today's `ingest/run.py` generalized to run under a Ray/Spark driver. SP11 fixes the **shape**; SP8 builds it.

---

## 6. Data flow

Nothing in the **query** path changes across any of these moves — that is the point.

```
QUERY (unchanged regardless of backend):
  answer(question, acl)
    └─ retriever.retrieve(Query)
         └─ vector_store.search(embedding, top_k, acl)   # acl.tenant_id already present
              ├─ pooled default:   ONE collection + MUST tenant_id filter   (today)
              └─ physical (proposed): route to namespace {base}:{corpus_ver}:{acl.tenant_id}
                                       — no shared index to mis-filter (fail-closed, §8)

INGEST (shape SP11 targets; SP8 implements):
  batch lane  (Ray/Spark backfill) ─┐
  stream lane (Kafka/CDC deltas)  ──┴─► embedder.embed_documents(...)   # API today | self-hosted (proposed)
                                        vector_store.upsert(chunks)     # incremental; lazy-creates namespace (D7)
                                        corpus_version bump → new generation takes new writes (D4)
```

The **only** code that differs between "pooled today" and "physical at 100×" is *inside* the concrete `VectorStore` provider and one `build_vector_store` branch. `core/pipeline.py`, `core/interfaces.py` (bar the noted seam), `retrieval/hybrid.py`, and every eval/guardrail path are byte-for-byte identical.

## 7. Config knobs (core/config.py)

All proposed knobs default to **today's behavior** — confirming this spec is a no-op at runtime (D8).

| Knob | Default | Purpose |
|------|---------|---------|
| `vector_store` (existing, extend the `Literal`) | `qdrant` | Add `"turbopuffer"` as an accepted value → serverless/physical store (D3). Unchanged default. |
| `tenant_isolation` (PROPOSED) | `pooled` | `pooled` = today's single-collection + filter; `physical` = namespace-per-tenant. Only honored by stores that support it; a store that can't do `physical` must **fail loudly at build**, never silently fall back (§8). |
| `embed_provider` (PROPOSED) | `api` | `api` = OpenAI-compatible (NIM/OpenAI) as today; `selfhosted` = local GPU-served embedder (D6). |
| `ingest_mode` (PROPOSED, SP8-owned knob, listed for completeness) | `batch` | `batch` = today's one-shot; `streaming` = CDC lane. SP8 owns the implementation and the final knob shape. |
| `corpus_version` source (existing, cache) | `.cache/corpus_version` | Reused as the **generation** id for routing (D4). No new knob — same counter. |

Existing knobs relied on unchanged: `qdrant_collection`, `pg_table`, `embed_base_url`, `embed_model`, `embed_dimension`.

## 8. Error handling — fail closed on the security path

- **Physical isolation must fail closed.** A tenant-routing store must **never** fall back to a shared/default collection when the tenant→namespace mapping is missing or a tenant id is empty — it **raises and fails the query**. This mirrors the standing rule in `docs/architecture.md:266` ("Never derive ACLContext from the prompt") and SP1's fail-closed filter stance: an isolation gap is a security failure, not a degradation.
- **`tenant_isolation=physical` on a store that can't do it fails at build time** in `core/registry.py`, not at query time — a `ValueError` at `build_vector_store`, the same pattern as the existing `raise ValueError(f"Unknown vector_store: …")` (`core/registry.py:46`). No silent pooled fallback.
- **Lazy namespace creation (D7) is create-if-absent, never read-through-shared.** A missing namespace for a *known, non-empty* tenant is created empty (first write) — it must not resolve to another tenant's namespace.
- **A missing `corpus_version` file** defaults to `"0"` (matches the cache spec's fresh-checkout behavior) — deterministic, never "latest wins."
- **Self-hosted embedder dimension mismatch** must fail at `ensure_collection` (dimension is config-driven, `interfaces.py:32`), not silently write wrong-width vectors.
- This spec introduces **no runtime path on confirm** (D8), so the primary error-handling contract is: *the future providers inherit fail-closed; the decision itself changes nothing.*

## 9. Testing (TDD) — minimal (narrative doc)

This slice ships no runtime code, so testing is deliberately light. The *tests belong to the slices that implement each provider* (SP8, and whoever writes the serverless/self-host providers). What SP11 pins for those future slices:

- **Isolation parity test (contract):** when a physical-isolation store lands, it must pass the *existing* cross-tenant isolation test (mirrors `tests/test_multitenant_isolation.py` and the cache spec's §10 cross-tenant test) — tenant A's write is invisible to tenant B — with the store in `physical` mode.
- **Fail-closed build test:** `build_vector_store` with `tenant_isolation=physical` against a store lacking support raises `ValueError` (no silent pooled fallback).
- **Routing test:** a fake routing store records that `search`/`upsert` addressed `{base}:{version}:{tenant}` for the request's `acl.tenant_id` — offline, no live backend (same fake-driven pattern as the rest of the suite).
- **No-op-on-confirm test:** with all proposed knobs at their defaults, the built components are identical to today's (`vector_store=qdrant`, `embed_provider=api`) — proves D8.

No new tests are required to *land this document*.

## 10. Files

**Create**
- `docs/superpowers/specs/2026-07-12-sp11-scale-narrative-design.md` — **this file** (the VDB-Decision).

**Modify (documentation only)**
- `README.md:318-331` — rewrite the "What I'd Do at 100× Scale" table into the designed-for-vs-aspirational architecture from §4, with the five Notion lessons and the Protocol/registry swap story. (Optionally add a scaling architecture diagram alongside the existing Mermaid.)

**Proposed for future slices (NOT created here — listed so reviewers see the seam):**
- `providers/vectorstores/turbopuffer_store.py` (serverless/physical store) + a `build_vector_store` branch in `core/registry.py:36` — *aspirational*, owned by the store-adoption slice.
- `providers/embedders/selfhosted.py` + a `build_embedder` branch in `core/registry.py:29` — *aspirational*.
- Streaming/batch ingest lanes — *aspirational*, owned by **SP8**.
- `core/config.py`: `tenant_isolation`, `embed_provider` knobs (§7) — added by the slice that first needs them, not by this doc.

## 11. Open questions / future hooks

- **`ensure_collection` seam (D7):** confirm **lazy-create-on-upsert** over widening the Protocol to `ensure_collection(dimension, tenant=None, generation=None)`. Lazy-create keeps the signature stable but hides creation cost in the write path; a widened signature is explicit but ripples to Qdrant/pgvector. Which does the user prefer?
- **Generation retention policy:** once routing writes to `{base}:{corpus_version}`, how long do old generations live (TTL vs. explicit drop after a re-embed backfill completes)? The cache uses TTL; the store may want explicit drop after backfill verification.
- **BM25 at 100×:** the sparse lane is an in-process pickled `rank-bm25` index (`ingest/run.py:95-97`). Physical per-tenant isolation of the *dense* store leaves the *sparse* store pooled. Future hook: OpenSearch/Elasticsearch BM25F with per-tenant routing — a new `SparseRetriever` provider, out of this decision.
- **Reranker/generator scaling** (GPU batching, NIM throughput) — named in the README rewrite as aspirational; no Protocol change needed (both are already swappable), so deferred to a future ops slice.
- **Serverless cost validation:** Notion's ~60% / ~90% / 600× are *their reported figures*, not independently verified here. Before adopting Turbopuffer-class storage, run a cost/latency A/B against our corpora under the SP5 eval gate (which owns the gate).
- **Cross-region / data-residency:** physical per-org collections make per-region placement tractable (an org's namespace pinned to a region). Out of scope; a natural follow-on once physical isolation exists.
