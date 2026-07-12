# VDB-Decision · Vector Database Selection — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Decision checkpoint before SP8 (physical-isolation / ingest-at-scale work). This is a **decision document**, not an implementation slice: the `VectorStore` Protocol (`core/interfaces.py:40`) already abstracts the store, so the production choice is a config-level switch. It depends on SP1 (verified tenant identity — the ACL scope must be trustworthy before isolation grade matters) and gates SP8 **only if** physical per-tenant isolation is chosen.

---

## 1. Context & problem

This slice fixes **no code defect**. The "problem" is a *decision that has never been made on the record*: the production vector database is implicitly locked to the Qdrant default, with no head-to-head evaluation, and the isolation grade (pooled metadata-filter vs physical namespace/collection-per-org) is undecided going into SP8. The table below is **verified current-state facts** (real file:line), each of which is a *condition forcing the decision* — not a bug to patch.

| # | Current-state fact (forcing function) | File:line (verified) |
|---|---|---|
| 1 | Production store is a single `Literal["qdrant","pgvector"]` switch defaulting to `qdrant`; no candidate beyond these two exists. | `core/config.py:37` |
| 2 | **Pooled (logical) multi-tenancy only**: one shared Qdrant collection (`qdrant_collection="rag_chunks"`) / one shared pg table (`pg_table="rag_chunks"`); all orgs co-reside, isolated solely by a `tenant_id` payload/column filter. No namespace/collection-per-tenant code exists anywhere (grep-confirmed). | `core/config.py:39,41`; `providers/vectorstores/qdrant_store.py:70`; `providers/vectorstores/pgvector_store.py:48` |
| 3 | ACL is enforced pre-similarity in **both** stores (Qdrant payload filter; pg `WHERE` before `ORDER BY <=>`) — so isolation is *correct* today, but *logical*, on a shared index. | `providers/vectorstores/qdrant_store.py:118`; `providers/vectorstores/pgvector_store.py:138-140`; `retrieval/acl.py:21,60` |
| 4 | The three ACL filter builders (`qdrant_filter`, `pg_where`, `acl_predicate`) are **independent re-implementations** of `ACLContext.allows()` — a store swap adds a fourth that could silently diverge, so a portable isolation contract is a prerequisite for switching. | `retrieval/acl.py:21,60,75`; `core/types.py:30` |
| 5 | pgvector's index is hardcoded `ivfflat WITH (lists=100)` — a fixed recall/latency operating point that does not scale to millions of vectors and pins pgvector's realistic ceiling. | `providers/vectorstores/pgvector_store.py:78-80` |
| 6 | The real store filters have **zero behavioral coverage offline** (only structure-checked; live tests self-skip) — evaluating a new candidate therefore has nothing to inherit; it needs an executable parity contract. | audit `docs/PRODUCTION_READINESS_AUDIT.md:556,559-561` |

**Tenancy model recap.** A *tenant* is an *org*. The target regime is **thousands of orgs × millions of vectors**, with meaningful **onboarding throughput** (new orgs created continuously). At that scale the isolation grade and the cost model (pay-per-uptime vs serverless/object-storage) stop being cosmetic and become the load-bearing choice.

### 1.1 Evaluation matrix (the heart of this decision)

Criteria × candidates. Latency/cost cells are deliberately **"to be measured against our corpus"** — this draft does not invent numbers. External-product capability cells marked *(to verify)* must be confirmed against current vendor docs before the decision is locked; the rest are given or established in-repo.

| Criterion | Qdrant (default, wired) | pgvector (wired) | Turbopuffer | Milvus / Vespa / LanceDB (noted) |
|---|---|---|---|---|
| **Isolation grade** | Logical (pooled filter) today; **physical** possible via collection-per-org | Logical (pooled filter) today; physical via schema/table-per-org (heavy) | **Physical by design** — namespace-per-tenant is the native unit | Milvus: partition/collection-per-tenant; Vespa: content-cluster; LanceDB: table/dataset-per-tenant *(to verify)* |
| **Scale (orgs × vectors; onboarding throughput)** | Strong; many collections has per-collection overhead → onboarding cost *(to verify at thousands of orgs)* | Weak past low-millions with `ivfflat lists=100`; table-per-org explodes DDL/onboarding | **Designed for many tenants**: cheap namespace creation, object-storage backing → high onboarding throughput *(to verify)* | Milvus/Vespa scale high, heavier ops; LanceDB embedded, single-node oriented |
| **Cost model** | Pay-per-uptime (self-host or managed cluster always on) | Pay-per-uptime (Postgres already running → marginal) | **Serverless / object-storage** — pay per stored bytes + query, cold tenants ≈ free *(to verify)* | Mostly pay-per-uptime; LanceDB = storage-only, no server |
| **p50/p95 latency, filtered queries** | to be measured against our corpus | to be measured against our corpus | to be measured (object-storage cold-start tail is the risk to quantify) | to be measured |
| **Native hybrid (dense + sparse)** | **Yes** — native sparse vectors + fusion | **No** native sparse; needs external BM25 (this repo's `BM25Retriever`) or `tsvector` bolt-on | *(to verify — assume external sparse needed)* | Milvus: yes; Vespa: yes (strong); LanceDB: FTS *(to verify)* |
| **Ops burden** | Self-host **or** managed cloud | Lowest *if* already operating Postgres; otherwise a full DB to run | **Managed only** — lowest ops, vendor dependency + data residency to assess | Milvus/Vespa: highest self-host burden; LanceDB: no server, app-embedded |

---

## 2. Goals

- Put a **defensible, criteria-driven decision on the record** for the production vector database, with an explicit decision framework and the discriminators that break ties.
- **Propose a default** (with rationale) and enumerate the **exact conditions that flip it**, so the decision is re-evaluable without re-litigating from scratch.
- State **WHEN** the decision must be made relative to SP8, and what makes it deferrable vs blocking.
- Define a **store-contract parity test** (offline-runnable) that any candidate must pass, so the isolation guarantee is portable across a store swap and cannot silently diverge from `ACLContext.allows()`.
- Preserve the existing architecture: any future store is a `providers/vectorstores/` impl wired **only** in `core/registry.py:build_vector_store`, selected by the `vector_store` config literal. No parallel structure.

## 3. Non-goals (deferred) — with the owning sub-project

- **Building a namespace/collection-per-tenant store impl and the ingest path that populates it** → **SP8** (physical isolation / ingest-at-scale). This spec decides *whether/when*; SP8 builds it *if* physical isolation is chosen.
- **Running the real store filters in CI (ephemeral service containers, skip-count assertion)** → **SP5** (eval/CI gate). This spec defines the parity *test*; SP5 owns *executing* it against live services in the pipeline.
- **Verified tenant identity (JWT-derived `tenant_id`/`acl_tags`)** → **SP1** (Security & tenancy). Already specced; a prerequisite, not part of this decision.
- **Org registry / self-service org creation / per-org onboarding API** → **SP1.5** (Org & corpus management). Onboarding *throughput* is an evaluation criterion here; the *mechanism* is SP1.5/SP8.
- **Sparse retriever choice and hybrid fusion tuning** → **SP4** (Retrieval correctness). Native-hybrid support is a *criterion* here, not a redesign of the retriever.
- **Cost/latency benchmarking harness against our corpus** → measurement task feeding this decision; owned jointly with **SP7** (observability/cost). This draft leaves those cells "to be measured."

---

## 4. Decisions (PROPOSED) — confirm/override on review

Each row leads with the best-practice option. **These are proposed for the user to confirm or override.**

| # | Decision | PROPOSED choice | One-line rationale |
|---|---|---|---|
| D1 | **Production default store** | **Keep Qdrant** | Already wired + tested, native dense+sparse hybrid, strong filtered-query performance, self-host *or* managed — no new system, lowest switching cost until a discriminator fires. |
| D2 | **Isolation grade (now)** | **Pooled logical filter** (status quo) | ACL is already enforced pre-similarity and correct; physical isolation is deferrable *by design* behind the Protocol until a compliance or scale trigger fires. |
| D3 | **Primary tie-breakers** | **Isolation grade, then scale/onboarding throughput** | These are the only criteria that force a physical-isolation store; latency/cost differentiate *within* a grade, not across the decision. |
| D4 | **Flip-to-physical candidate** | **Turbopuffer** (namespace-per-tenant, object-storage) | If a hard trigger fires, the physical-per-tenant + serverless cost model at thousands-of-orgs scale is Turbopuffer's native shape — cheaper cold tenants and higher onboarding throughput than many-collections-on-Qdrant *(to verify against corpus + vendor docs)*. |
| D5 | **"One-fewer-system" alternative** | **pgvector** — only if already operating Postgres at the needed scale | Marginal ops cost when Postgres is already run; ruled out at target scale by `ivfflat lists=100` ceiling and no native sparse. |
| D6 | **Milvus / Vespa / LanceDB** | **Note, do not adopt now** | Milvus/Vespa add heavy self-host ops for no decisive edge over Qdrant at our stage; LanceDB is embedded/single-node — revisit only if a specific need (e.g. Vespa's ranking, LanceDB's zero-server embed) emerges. |
| D7 | **Portable isolation contract** | **Store-contract parity suite** (§9) every impl must pass | Prevents a fourth filter builder from silently diverging from `ACLContext.allows()` on a store swap (Fact #4). |
| D8 | **Decision timing** | **Blocking before SP8 iff physical isolation is chosen; otherwise deferrable** | The load-bearing sentence: pooled-filter status quo needs no store change, so SP8 can proceed on Qdrant; a physical-isolation mandate must resolve *before* SP8 builds ingest, because it changes the collection/namespace topology at write time. |

**Conditions that flip D1/D2 (pooled Qdrant → physical Turbopuffer), any one sufficient:**
1. A **compliance / contractual mandate** for physical per-tenant data separation (not satisfiable by a shared-index filter).
2. Scale reaches **thousands of orgs × millions of vectors** with **high onboarding throughput**, where many-collections-on-Qdrant per-collection overhead or pay-per-uptime cost becomes the bottleneck (to be measured).
3. A large population of **cold/low-traffic tenants** where serverless/object-storage economics dominate pay-per-uptime (to be measured).

If none fire, **stay on pooled Qdrant** and defer.

---

## 5. Architecture & components

No new architecture is introduced by *this decision*. The point is that the existing abstraction already makes the choice a config switch:

- **`VectorStore` Protocol** (`core/interfaces.py:40-49`) — the sole contract: `ensure_collection`, `upsert`, `search(embedding, top_k, acl)`, `count`. ACL is a **required** search argument, applied pre-similarity. Any candidate must satisfy exactly this.
- **Concrete impls** live in `providers/vectorstores/` (today: `qdrant_store.py`, `pgvector_store.py`). A future physical-isolation store (e.g. `turbopuffer_store.py`, or a namespace-per-tenant variant) is **one more class here** — nothing in the pipeline changes.
- **Registry** (`core/registry.py:build_vector_store`) is the **only** place a concrete store is named; it dispatches on `settings.vector_store`. Adding a candidate = one new literal value + one branch here.
- **ACL builders** (`retrieval/acl.py`) — a new physical-isolation store still needs an intra-namespace ACL filter for `acl_tags` (tenant is the namespace; tags remain within it). It adds a builder that MUST pass the §9 parity suite.

**Single-purpose units, if/when SP8 builds the flip:**
1. `providers/vectorstores/<candidate>_store.py` — implements `VectorStore`; maps `tenant_id → namespace/collection`, applies `acl_tags` filter within it.
2. `retrieval/acl.py` — one new builder for that store's filter dialect (if needed), modelling `allows()` exactly.
3. `core/registry.py` — one branch; `core/config.py` — one literal value + isolation-mode knob (§7).

## 6. Data flow

The retrieval path is **unchanged** by the store choice — that is the whole design point:

```
query → Retriever → VectorStore.search(embedding, top_k, acl)  ← acl REQUIRED, pre-similarity
                                    │
              ┌─────────────────────┴─────────────────────┐
     pooled (logical)                            physical (per-tenant)
     ─────────────────                           ────────────────────
     one shared collection/table                 namespace/collection = tenant_id
     filter: tenant_id == acl.tenant_id           select namespace(acl.tenant_id)
             AND (open OR tag-overlap)             then filter: acl_tags overlap
     (qdrant_filter / pg_where today)             (new builder; §9 parity-tested)
                                    │
                             list[ScoredChunk]  (identical shape either way)
```

Onboarding flow differs by grade: pooled = write rows with a new `tenant_id` (no DDL); physical = create a namespace/collection per new org (the onboarding-throughput criterion). Both surface through the same `ensure_collection`/`upsert` contract.

## 7. Config knobs (`core/config.py`)

Only D-driven, minimal additions. **All PROPOSED.**

| Knob | Default | Purpose |
|---|---|---|
| `vector_store` (extend literal) | `"qdrant"` | Current: `qdrant`\|`pgvector`. **Proposed:** add candidate values (e.g. `turbopuffer`) *only when* SP8 builds the impl — not before, to avoid a dead branch. |
| `tenant_isolation` | `"pooled"` | `pooled` \| `physical`. Selects logical filter vs namespace/collection-per-tenant. Fail-closed: an unknown value must raise at registry build, never default to `pooled` silently. |
| `turbopuffer_api_key` | `""` | (Only if D4 adopted) managed-service credential; empty in dev. |
| `turbopuffer_region` | `""` | (Only if D4 adopted) data-residency selection — a compliance input, must be explicit. |

No config is added by the *decision itself*; this table specifies what SP8 would add *if* the flip is approved. The default row (`vector_store="qdrant"`, implicit `tenant_isolation="pooled"`) is the status quo.

## 8. Error handling — fail closed on the security path

- **Registry build** (`build_vector_store`): an unrecognised `vector_store` or `tenant_isolation` value raises `ValueError` (matches the existing `raise ValueError(f"Unknown vector_store: …")` pattern) — never falls back to a default store or a shared namespace.
- **Namespace selection (physical mode):** a blank/missing `tenant_id` in the `ACLContext` must **raise, not** resolve to a shared/empty namespace (mirrors the SP1 blank-tenant fail-closed rule; a `""` tenant is a real distinct namespace and must be rejected upstream).
- **ACL filter parity:** any new store's filter builder that fails the §9 parity suite **blocks the swap** — an isolation regression is a build failure, not a runtime surprise.
- **Managed-service outage (Turbopuffer/managed Qdrant):** `search` errors propagate as a failure (empty/no-context answer or explicit error) — **never** a silent unfiltered or cross-tenant result. No fail-open.
- **Cold-namespace / missing-namespace read:** returns empty results for that tenant, never another tenant's namespace.

## 9. Testing (TDD) — offline-testable behaviors

The honest TDD fit is a **store-contract parity suite** — the same behavioral tests, parameterised across every `VectorStore` impl, so isolation is proven per-store, not just structure-checked (closes Facts #4/#6). Offline-runnable via ephemeral/embedded backends (`QdrantClient(":memory:")`, pgvector container for its own tier); the flip candidate joins the same suite when built.

Concrete behaviors each impl MUST satisfy:
1. **Cross-tenant zero-leak:** a `tenant_b` chunk engineered (higher raw similarity) to out-rank every `tenant_a` chunk returns **zero rows** for a `tenant_a` ACL. (The exact audit-recommended test — proves the *filter executes*, not just parses.)
2. **Open-chunk visibility:** a no-tag chunk is visible to any caller in its tenant; a tagged chunk is visible only with overlapping tags — asserted through the real store, matching `allows()`.
3. **No-tag caller sees only open chunks:** empty caller tags ⇒ tagged chunks excluded (the `MatchAny` / `&&` short-circuit correctness).
4. **Parity oracle:** for a randomized corpus + ACL, `store.search(...)` membership == the set `acl_predicate(acl)` accepts — every store agrees with `ACLContext.allows()`.
5. **Physical-mode namespace routing (when built):** a write under tenant A and a read under tenant B touch **different namespaces**; blank tenant **raises**.
6. **Registry fail-closed:** unknown `vector_store` / `tenant_isolation` raises `ValueError`; no default fallback.

CI *execution* of live-service variants is **SP5's** responsibility (§3); this spec defines the tests and the offline/embedded variants.

## 10. Files

**Create (now):**
- `docs/superpowers/specs/2026-07-12-vdb-decision-design.md` — this decision record.

**Create (only if SP8 approves the flip to physical/Turbopuffer — not now):**
- `providers/vectorstores/turbopuffer_store.py` — `VectorStore` impl (or a namespace-per-tenant Qdrant variant).
- `tests/test_vectorstore_parity.py` — the parameterised parity suite from §9 (a scaffold may land earlier under SP5 to cover the two existing stores).

**Modify (only if the flip is approved):**
- `core/config.py` — extend `vector_store` literal; add `tenant_isolation` + any managed-service knobs (§7).
- `core/registry.py` — one branch in `build_vector_store` for the new store / isolation mode.
- `retrieval/acl.py` — one new filter builder (if the store's dialect requires it), parity-tested.

**No code is created or modified by the decision itself.** This is a checkpoint document.

## 11. Open questions / future hooks

- **Trigger measurement:** the p50/p95-under-filter, per-collection onboarding overhead, and cold-tenant cost cells are all "to be measured against our corpus" — a small benchmarking task (with SP7) must fill them before D1/D4 is *locked*, not just proposed.
- **Vendor-capability verification:** every *(to verify)* cell (Turbopuffer native hybrid, namespace-creation cost, Qdrant many-collections overhead, LanceDB FTS) must be confirmed against current vendor docs before locking — do not lock on memory.
- **Naming reconciliation:** SP1's spec (`2026-07-12-sp1-security-tenancy-design.md:34,200-201`) references physical isolation under both "VDB-Decision / SP11" and "before SP8 ingest." This document aligns to the program phrasing — **the physical-isolation build is SP8**; the "SP11" label there is the same work and should be updated to SP8 for consistency.
- **Hybrid-parity across stores:** if the flip candidate lacks native sparse, the existing `BM25Retriever` (`providers/sparse/bm25.py`) must carry sparse for that store — confirm its tenant-partitioned isolation is itself parity-tested (currently dead on the request path per audit).
- **Migration path:** a pooled→physical switch is a re-ingest, not an in-place migration — SP8 must own a backfill that re-partitions existing pooled data into per-tenant namespaces without a cross-tenant window.
- **Revocation/onboarding coupling:** physical isolation makes per-org teardown trivial (drop namespace) — a potential hook for the SP1.5 org lifecycle (de-provisioning) worth noting when D4 is evaluated.
