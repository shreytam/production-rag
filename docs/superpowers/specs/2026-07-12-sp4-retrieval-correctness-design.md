# SP4 · Retrieval Correctness — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 0 (risk-ordered). Fourth slice, after SP3 (Compliance). Assumes SP1 (auth/tenancy) and SP2 (guardrails) are in place — this slice touches only the retrieval path and does not re-open identity or safety concerns.

---

## 1. Context & problem

The "hybrid" retriever is the headline capability of this system, but on the **deployed request path it silently degrades to dense-only** and the tenant-partitioned BM25 ACL logic is dead code. The eval harness cannot catch this because it exercises a *different* build call. Several adjacent retrieval-correctness defects compound the recall loss. All line numbers below were verified by reading the current source.

| # | Defect | Location | Reality (verified) |
|---|--------|----------|--------------------|
| 1 | Deployed API runs **dense-only**, not hybrid | `app/api.py:32` | `build(version="full", dataset=None)`. `None` flows to `_load_bm25` (`core/pipeline.py:44`) which returns `None`; `build` then falls back to `build_sparse_retriever(s)` (`core/pipeline.py:218`) → a **fresh empty** `BM25Retriever()` (`core/registry.py:52`) that was never `.index()`-ed. |
| 2 | Empty BM25 → `search()` returns `[]` | `providers/sparse/bm25.py:60-61` | `acl.tenant_id not in self._indices` (indices is `{}`) → `return []`. RRF then fuses dense against an empty list (`retrieval/hybrid.py:51-52`), so the sparse arm contributes nothing and per-tenant BM25 isolation is never exercised. |
| 3 | The demo path has the same latent bug | `app/demo.py:32` | `build(..., dataset=dataset or None)` → empty/omitted dataset repeats defect #1. |
| 4 | No loud signal when `version='full'` runs with an empty sparse index | `core/pipeline.py:217-220` | `build` accepts a non-indexed sparse retriever silently; a mis-wired deploy looks healthy. |
| 5 | Chunker drops trailing-overlap tokens on oversized paragraphs | `ingest/chunking.py:44-57, 147-148` | `_token_chunks_from_paragraph` slices oversized paragraphs into back-to-back `max_tokens` windows **with no overlap**, and the packer's `window = window[:max_tokens]` trim (`:148`) discards the tail. Recall loss scales with document length. |
| 6 | Dead first-pass chunking loop | `ingest/chunking.py:92-132` | A full `for i, slice_tokens in enumerate(...)` loop builds `chunks`, then line 133 does `chunks = []` and redoes the work with a `while` loop. The first pass is pure waste (and confuses maintainers via the apologetic comments at `:130-132`). |
| 7 | Context assembly tokenizer not aligned to gen model | `core/context_assembly.py:16-24` | Budgeting hardcodes tiktoken `cl100k_base`. The default `gen_model` is `meta/llama-3.3-70b-instruct` (`core/config.py:53`), whose tokenizer differs — the 4000-token budget (`core/config.py:89`) is mis-measured, risking over/under-fill and truncated context at the model boundary. |
| 8 | RRF tie-handling is order-dependent (unstable) | `core/rrf.py:35` | `sorted(fused.items(), ...)` breaks equal RRF scores by dict-insertion order — a subtle nondeterminism across runs and a hard-to-reproduce ranking. Cross-source dedup is by `chunk_id` only (`:32`), fine, but ties need a deterministic key. |
| 9 | Reranker input contract is duplicated, not shared (cleanup) | `providers/rerankers/nim_rerank.py:44,67-72,79` | Both rerankers already early-return `[]` on empty input and `nim` already bounds-checks `idx` (`:79`) — so this is **cleanup, not a live correctness bug**. The real issues: `_score` is computed twice per item, and there is no single tested contract for degenerate/malformed remote responses shared across the two impls. |

**Why eval doesn't catch #1/#2:** `eval/run_eval.py:76` and `eval/ragas_adapter.py:131` call `build(version=version, dataset=dataset, ...)` **with a real dataset name**, so `_load_bm25` finds the persisted `.cache/bm25_<dataset>_<store>.pkl` (written at `ingest/run.py:95-97`) and hybrid works. The API/demo pass `dataset=None`, taking the broken branch. The two paths diverge exactly where it isn't measured.

## 2. Goals

- The **deployed API and demo run true hybrid retrieval**: the sparse (BM25) arm is a populated, tenant-partitioned index loaded from the persisted artifact, so RRF fuses two real rankings and per-tenant BM25 ACL isolation is live on the request path.
- **Fail loud, fail closed on a misconfigured hybrid build**: when `version='full'`, a sparse retriever with zero indexed chunks is a hard error (configurable to warn), not a silent dense-only fallback.
- **Token-conserving chunking**: oversized paragraphs preserve trailing overlap so no tokens are silently dropped; the dead first-pass loop is removed.
- **Budget alignment**: context assembly measures tokens with a tokenizer that matches (or conservatively over-estimates) the active gen model, parameterized via config.
- **Deterministic fusion**: RRF ties break on a stable, documented key; cross-source dedup semantics are explicit and tested.
- **Uniform reranker empty/degenerate-input contract**: empty candidates → `[]`; malformed remote responses degrade safely and are tested offline.
- An **integration test asserts the API-built pipeline's sparse retriever is non-empty** and that hybrid actually fuses two arms — closing the eval/deploy gap permanently.

## 3. Non-goals (deferred) — owned elsewhere

- **Ingest robustness, incremental / delta re-embedding, BM25 rebuild orchestration** → **SP8 (Ingest Robustness)**. SP4 loads the *existing* persisted BM25 artifact and hardens the read path; it does not redesign how/when it is (re)built.
- **Authentication / identity extraction** (X-Tenant-Id → verified claim) → **SP1 (Security & Tenancy)**. SP4 consumes the `ACLContext` SP1 produces; it does not touch auth.
- **Guardrails / injection / groundedness** → **SP2 (Guardrail Correctness)**. Untouched here.
- **PII / compliance redaction of retrieved text or traces** → **SP3 (Compliance)**.
- **Vector-store schema, embedding-dimension, or ANN-index tuning** → **SP8**. SP4 does not alter dense storage.
- **A global request-id / exception handler** → **SP9 (Ops)**. SP4's error handling is scoped to the retrieval build + reranker.

## 4. Decisions (PROPOSED)

> These are **proposed** for the user to confirm or override on review. Each leads with the best-practice option.

| # | Decision (PROPOSED) | Rationale |
|---|---------------------|-----------|
| D1 | **Keep the corpus as an explicit build argument** (the existing `dataset` param, renamed `corpus` for clarity) and default it *at the API/demo call site* from a new config knob `active_corpus`. One corpus *value*, two *sources*: config for the deploy, argument for eval. | Deploy becomes hybrid via one env var **without** regressing the eval path, which already passes its per-dataset corpus explicitly. A pure config-global would break eval (it never sets `active_corpus`) and cannot serve two corpora in one process because `get_settings()` is `@lru_cache`d — see §11. |
| D2 | **Fail closed by default** on empty sparse index when `version='full'`: raise `HybridIndexError` unless `hybrid_require_sparse=false` (then log a loud `WARNING` and proceed dense-only). | A "hybrid" pipeline that isn't hybrid is a correctness incident, not a soft degrade. Escape hatch stays for genuine dense-only deploys. |
| D3 | **Load BM25 via a small `SparseIndexLoader` seam** (Protocol in `core/interfaces.py`, impl in `providers/sparse/`), not ad-hoc `pickle.load` inside `pipeline.py`. | Keeps `pipeline.py` provider-free; makes the artifact source swappable (pickle today, a store later) and testable with a fake. |
| D4 | **Move BM25 artifact load out of `pipeline._load_bm25`** into the loader, keyed by the passed `corpus` argument; `build` calls `registry.build_sparse_retriever(s, corpus=corpus)` which returns a *loaded* retriever when an artifact for that corpus exists. | Consolidates concrete-class knowledge in `registry`, per the codebase's single-factory rule; the corpus stays an argument so eval and deploy share one mechanism. |
| D5 | **Chunker: overlap-preserving paragraph split.** `_token_chunks_from_paragraph` emits windows that carry `overlap` tokens of overlap between adjacent slices; the packer no longer hard-trims tail tokens. | Directly fixes recall loss; conserves every token across the split boundary. |
| D6 | **Delete the dead first-pass chunking loop** (`ingest/chunking.py:92-132`), keep the single `while`-based pass. | Removes ~40 lines of confusing dead code; zero behavior change beyond the fix in D5. |
| D7 | **Parameterize the assembly tokenizer** via `context_tokenizer` config; resolve model→encoding through a small map with a **conservative fallback** (`o200k_base`/`cl100k_base`) and a safety margin `context_token_safety_margin`. | Aligns budgeting to the real model; the margin guarantees we under-fill rather than overflow the model's context window. |
| D8 | **RRF deterministic tie-break**: sort by `(-rrf_score, chunk_id)`. | Stable, reproducible ranking independent of insertion order; `chunk_id` is a total order. |
| D9 | **RRF cross-source dedup stays keyed on `chunk_id`; keep first-seen chunk object, union component scores** (already the behavior) — make it explicit + tested, and record all contributing sources. | Confirms intended semantics; prevents a future regression from silently changing it. |
| D10 | **Shared reranker input-normalization helper** `normalize_candidates()`: empty → `[]`; drop out-of-range indices; single `_score` computation. Both `local` and `nim` rerankers route through it. | One tested contract for degenerate input; removes duplicate/unsafe index handling in `nim_rerank`. |
| D11 | **BM25 artifact integrity check on load**: verify it unpickles to a `BM25Retriever` with ≥1 tenant index, else treat as empty (→ D2 policy). | A truncated/corrupt pickle must not masquerade as a healthy index. |

## 5. Architecture & components

Follows the existing pattern: **Protocols in `core/interfaces.py`, concrete impls in `providers/`, wired only in `core/registry.py`.** No parallel structure.

**New Protocol — `SparseIndexLoader` (`core/interfaces.py`):**
```
class SparseIndexLoader(Protocol):
    def load(self, corpus: str, store: str) -> SparseRetriever | None: ...
```
Returns a fully-indexed `SparseRetriever` for the corpus, or `None` when no artifact exists.

**Concrete impl — `providers/sparse/pickle_loader.py :: PickleSparseIndexLoader`:**
- Resolves `.cache/bm25_{corpus}_{store}.pkl` (same path `ingest/run.py` writes).
- Unpickles; runs the D11 integrity check (`isinstance(obj, BM25Retriever)` and `len(obj._indices) >= 1`); returns `None` on miss/corruption (loud `WARNING` on corruption).

**`core/registry.py` — `build_sparse_retriever(settings, corpus=None)`:**
- If `corpus` truthy → attempt `PickleSparseIndexLoader().load(corpus, settings.vector_store)`; return the loaded retriever if non-empty.
- Else / on miss → return a fresh empty `BM25Retriever()` (unchanged) — caller decides policy (D2).
- Registry remains the only module naming `BM25Retriever` / `PickleSparseIndexLoader`.

**`core/pipeline.py :: build(version, corpus=None, ...)`:**
- Drop `_load_bm25`. The `dataset` param is renamed `corpus` (a thin back-compat alias may be kept if any caller still passes `dataset=`). For `version='full'`: `sparse = build_sparse_retriever(s, corpus=corpus)`; then **assert non-empty per D2** (`_assert_hybrid_ready(sparse, s)` → raise `HybridIndexError` or warn).
- **Eval** (`eval/run_eval.py:76`, `eval/ragas_adapter.py:131`) keeps passing its per-dataset corpus as the argument — unchanged behavior, the reference path stays green.
- **API/demo** pass `corpus=s.active_corpus` at the call site (D1), so a single env var makes the deploy hybrid.

**`retrieval/hybrid.py`:** unchanged in shape; benefits from a real sparse arm. RRF call unchanged.

**`core/rrf.py`:** D8 tie-break + D9 explicit dedup/component-union; single-purpose, no interface change.

**`core/context_assembly.py`:** `_encoder()` becomes `_encoder(name)` (config-driven, still `lru_cache`d per name); `count_tokens`/`assemble_context` take/read the configured tokenizer + safety margin. `GroundedGenerator` passes `settings.context_tokenizer` through (via the existing `token_budget` wiring point).

**`ingest/chunking.py`:** D5 overlap-preserving split + D6 dead-loop removal. Pure functions, deterministic IDs unchanged.

**`providers/rerankers/`:** new `_common.py :: normalize_candidates()`; both impls import it. No interface change (`Reranker` Protocol untouched).

**New exception — `core/pipeline.py :: HybridIndexError`** (or `core/errors.py` if one exists).

## 6. Data flow

**Request path (fixed):**
```
env: ACTIVE_CORPUS=hotpotqa
  → get_pipeline() → build(version="full", corpus=s.active_corpus)   [app/api.py, app/demo.py]
    → build_sparse_retriever(s, corpus="hotpotqa")     [core/registry.py]
      → PickleSparseIndexLoader.load("hotpotqa","qdrant")
        → .cache/bm25_hotpotqa_qdrant.pkl → BM25Retriever (N tenant indices)
    → _assert_hybrid_ready(sparse, s)  # fail closed if empty & require_sparse
    → HybridRetriever(embedder, store, sparse, reranker)
  (eval takes the SAME path but passes corpus="hotpotqa" as an explicit argument)
Query:
  q → dense.search(acl)  ┐
  q → sparse.search(acl) ┘→ RRF(k, deterministic ties) → top fuse_window
      → reranker.rerank(normalize_candidates(...)) → assemble_context(tokenizer)
      → grounded.generate
```

**Ingest path (unchanged externally):** `chunk_document` now conserves overlap tokens; BM25 pickle still written by `ingest/run.py`.

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|------|---------|---------|
| `active_corpus` | `""` | Corpus the **API/demo** pass as the `corpus` build argument (deploy-side default only; eval passes its own corpus explicitly). Empty ⇒ no artifact ⇒ D2 policy applies. |
| `hybrid_require_sparse` | `True` | When `version='full'` and the sparse index is empty: `True` → raise `HybridIndexError` (fail closed); `False` → loud WARNING + dense-only. |
| `sparse_index_dir` | `".cache"` | Directory holding `bm25_{corpus}_{store}.pkl`. Matches current ingest write path. |
| `context_tokenizer` | `"auto"` | Encoding for context budgeting. `"auto"` ⇒ resolve from `gen_model` via the model→encoding map; explicit tiktoken encoding name overrides. |
| `context_token_safety_margin` | `0.10` | Fractional headroom subtracted from `context_token_budget` to absorb tokenizer mismatch (under-fill, never overflow). |
| `chunk_overlap` | `32` | (Surface existing `overlap` default as a knob.) Tokens of overlap preserved across every split, including within oversized paragraphs. |

## 8. Error handling — explicit; security-critical paths fail closed

- **Empty/absent sparse index on a `full` build (D2/D11):** default **fail closed** — raise `HybridIndexError` at build time so a mis-wired deploy fails to start rather than silently serving dense-only (which also silently disables the per-tenant BM25 ACL check — a correctness *and* isolation concern). Downgrade to WARNING only when `hybrid_require_sparse=false` is explicitly set.
- **Corrupt/truncated pickle:** loader returns `None` + logs a WARNING with the path; build then applies D2 policy. A corrupt artifact never partially loads.
- **BM25 `search` on unknown tenant:** already returns `[]` (`bm25.py:60-61`) — correct fail-closed tenant isolation; retained and covered by a regression test.
- **Reranker degenerate input:** empty candidates → `[]`; a malformed remote NIM response (missing/short `rankings`, out-of-range `index`) drops the bad entries and returns the valid remainder — never raises into the request path, never emits an out-of-bounds chunk.
- **Tokenizer resolution miss:** unknown `gen_model` under `"auto"` → fall back to the most conservative encoding + apply the safety margin (under-fill), and log once. Budgeting never crashes generation.
- **Chunker:** empty/whitespace doc → `[]` (unchanged); overlap logic never produces a chunk exceeding `max_tokens`.

## 9. Testing (TDD) — concrete, offline behaviors

All tests run with fakes (no network, no models), consistent with `tests/test_pipeline_integration.py` / `tests/_fakes.py`.

1. **API pipeline is truly hybrid (the headline regression).** With `active_corpus` set and a fixture BM25 pickle present, build via the API entrypoint and assert `pipeline.retriever.sparse` has ≥1 tenant index (`len(sparse._indices) >= 1`) and that a query returns a fused result whose `component_scores` include both `dense` and `sparse` sources. Fails on today's code.
2. **Fail-closed on empty sparse.** `build(version="full")` with no artifact + `hybrid_require_sparse=True` raises `HybridIndexError`; with `False` it logs a WARNING and returns a dense-only-capable pipeline.
3. **Loader integrity.** Corrupt bytes at the artifact path → `load()` returns `None` + WARNING; a valid pickle with tenants → returns a populated retriever.
4. **Token conservation (chunker).** For a synthetic doc of a single oversized paragraph, assert the multiset of tokens across all chunk texts ⊇ the source tokens (no trailing loss), and every adjacent chunk pair shares exactly `chunk_overlap` boundary tokens. Assert no chunk exceeds `max_tokens`.
5. **Dead-loop removal is behavior-preserving-plus.** Golden test: chunk counts/IDs for normal (non-oversized) docs unchanged vs. a captured baseline; oversized-doc case now retains overlap.
6. **Tokenizer alignment.** With `context_tokenizer="auto"` and a Llama `gen_model`, `count_tokens` uses the mapped encoding; assembled context stays within `budget * (1 - safety_margin)`. Explicit override respected.
7. **RRF determinism.** Two rankings crafted to produce tied RRF scores fuse to an identical, `chunk_id`-ordered result across repeated shuffled inputs.
8. **RRF cross-source dedup.** A chunk present in both dense and sparse appears once, with both sources in `component_scores` and summed RRF contribution.
9. **Reranker contracts.** Empty candidates → `[]` (both impls). Fake NIM response with a missing/out-of-range `index` → those entries dropped, valid ones ranked; no exception, no out-of-bounds chunk.
10. **Tenant isolation on the request path.** Extend the existing isolation test so the *hybrid* (non-empty BM25) path proves tenant A never receives tenant B's chunks from the sparse arm.

## 10. Files

**Create**
- `providers/sparse/pickle_loader.py` — `PickleSparseIndexLoader` (+ integrity check).
- `providers/rerankers/_common.py` — `normalize_candidates()` helper.
- `tests/test_retrieval_hybrid_wiring.py` — tests 1–3, 10.
- `tests/test_chunking_conservation.py` — tests 4–5.
- `tests/test_context_assembly_tokenizer.py` — test 6.
- `tests/test_rrf.py` — tests 7–8 (if not already present; extend otherwise).
- `tests/test_reranker.py` — extend with test 9.

**Modify**
- `core/interfaces.py` — add the `SparseIndexLoader` Protocol.
- `app/api.py` — `build(version="full", corpus=s.active_corpus)` (drop `dataset=None`).
- `app/demo.py` — pass `corpus=dataset or s.active_corpus` at the call site.
- `core/pipeline.py` — remove `_load_bm25`; rename `dataset`→`corpus`; `build` uses `build_sparse_retriever(s, corpus=corpus)` + `_assert_hybrid_ready`; add `HybridIndexError`.
- `core/registry.py` — `build_sparse_retriever(settings, corpus=None)` loads via `PickleSparseIndexLoader` when `corpus` is set.
- `core/config.py` — new knobs (§7).
- `core/rrf.py` — deterministic tie-break + explicit dedup/component union.
- `core/context_assembly.py` — parameterized tokenizer + safety margin.
- `generation/grounded_generator.py` — thread `context_tokenizer` through the budget wiring.
- `ingest/chunking.py` — overlap-preserving split; delete dead first-pass loop.
- `providers/rerankers/local_cross_encoder.py`, `providers/rerankers/nim_rerank.py` — route through `normalize_candidates()`.
- `eval/run_eval.py:76`, `eval/ragas_adapter.py:131` — rename the passed kwarg `dataset=`→`corpus=` (behavior unchanged; a back-compat alias avoids even this if preferred). The reference path stays green.

## 11. Open questions / future hooks

- **Eval vs. deploy corpus source — resolved (D1).** The corpus is a build *argument*, defaulted from `active_corpus` only at the API/demo call site. Eval keeps passing its per-dataset corpus, so the reference path is unchanged; the deploy path becomes hybrid via one env var. This deliberately avoids a config-global `active_corpus`, which would (a) trip the D2 fail-closed check on the eval path (eval never sets it) and (b) be unable to build pipelines for two corpora in one process because `get_settings()` is `@lru_cache`d. Left here only to flag for the reviewer that the `dataset`→`corpus` rename touches eval call sites.
- **Multi-corpus deploys.** `active_corpus` is a single deploy-side default; the underlying `corpus` argument already supports per-build selection. A corpus→index registry and per-request corpus selector for serving several corpora at once is deferred to SP8; the `SparseIndexLoader` seam already supports it.
- **Exact Llama tokenizer.** `"auto"` maps to the closest tiktoken encoding with a safety margin, not Llama's true SentencePiece tokenizer. A precise HF-tokenizer path (optional dependency) is a future hook if budget precision proves insufficient; the conservative margin is the safe default until then.
- **BM25 artifact freshness.** SP4 loads whatever `ingest/run.py` wrote; detecting a stale index vs. the current dense collection (version/hash check) belongs to SP8.
- **RRF weighting.** Current RRF weights dense and sparse equally. Per-source weights are a plausible future knob but out of scope for a correctness slice.
