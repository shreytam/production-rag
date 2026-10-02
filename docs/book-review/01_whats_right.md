# Artifact 1 — What Your Codebase Gets Right (Aligned with *Hands-On RAG for Production*)

> **Codebase:** `Production RAG` · **Book:** *Hands-On RAG for Production* (Mendelevitch & Bao, O'Reilly)
> Every claim below was verified against actual source files. Chapter/page references point into the book.

**Verdict up front:** this codebase independently implements most of the book's production playbook — often matching it technique-for-technique. The gaps that remain are mostly operational (streaming, rate limiting, feedback loop), not architectural. Details in Artifacts 2 and 3.

---

## 1. Architecture — the book's two-flow blueprint, implemented cleanly

**Book (Ch. 1 p. 4, Ch. 2 pp. 24–27):** RAG = an **ingestion flow** (parse → chunk → embed → index) and a **query flow** (rewrite → retrieve → rerank → generate), kept separate.

| Book principle | Your implementation | Verdict |
|---|---|---|
| Two separated flows | `ingest/` (run.py, worker.py, incremental.py) vs `core/pipeline.py` query path — physically separate packages | ✅ |
| Swappable layers, no framework lock-in (Ch. 5 DIY-vs-platform: keep control where it differentiates) | Every component sits behind a `Protocol` in `core/interfaces.py`; `core/registry.py` is the *only* module naming concrete implementations. Swapping NIM↔OpenAI is one env var (`llm_base_url`) | ✅ textbook |
| One place reads configuration | `core/config.py` is the single knob panel; nothing else reads env directly; pydantic validators **fail fast in prod** (missing `jwt_secret`, dev signer enabled in prod ⇒ constructor raises) | ✅ exceeds book baseline |
| Versioned, reproducible stack (Ch. 1 LLMOps table: version prompts/models/embeddings) | Prompt text lives in `generation/prompts.py`; `_PROMPT_VERSION = "v1"` stamped on contextual prefixes (`ingest/incremental.py`); manifests record chunk hashes | ✅ |

---

## 2. Retrieval — exactly the book's two-stage design

**Book (Ch. 3 pp. 75–82):** Stage 1 = high-recall candidate generation (hybrid dense + BM25, fused with RRF or weighted scores); Stage 2 = precision reranking with a transformer cross-encoder. *"Stage-2 quality is capped by stage-1 recall."*

Your `HybridRetriever` (`retrieval/hybrid.py`) is this design, step for step:

- Dense (Qdrant ANN/HNSW) **and** sparse BM25 candidates fetched per query, each top-`retrieve_top_k=20`.
- Fused with `core/rrf.py` reciprocal rank fusion (`rrf_k=60`) — the book's recommended fusion method because it needs no score normalization across incomparable scales (p. 77).
- Top `fuse_window=40` reranked by a cross-encoder down to `rerank_top_n=8`. The default local model, **BAAI/bge-reranker-v2-m3**, is literally the model used in the book's own reranking code walkthrough (Ch. 3 p. 82, Table 3-2).
- The NIM reranker option (`providers/rerankers/nim_rerank.py`) mirrors the book's OSS-vs-managed reranker trade-off table.

**ACL scoping done the secure way.** Book (Ch. 4 p. ~118): *"wire a permission-based filter into the query flow — only chunks authorized for the querying user reach the LLM."* You do this at the store layer (`retrieval/acl.py`): the Qdrant filter enforces `tenant_id` AND (`acl_open` OR tag-overlap) as **pre-similarity MUST clauses**, and collection scoping rides in the same filter. The in-memory predicate delegates to a single source of truth (`ACLContext.allows()`). Candidates are tenant-scoped **before fusion**, never filtered afterward — precisely the book's requirement, and stronger than most demo stacks.

**Query rewriting.** Book (Ch. 2 p. 25): rewrite user intent into a retrieval query, keep the original for generation. `core/pipeline.py` SP12 does exactly this split — `rewriter.rewrite()` output feeds retrieval + cache keys only; generation always sees the raw question.

---

## 3. Ingestion — restartable, incremental, idempotent

The book hammers three principles (Ch. 3 pp. 67–74): design pipelines to be **restartable not just runnable**, prefer **incremental updates over full re-indexing**, use **change detection** to touch only affected chunks. `ingest/incremental.py` implements all three:

- blake2b content hashes per chunk detect new/changed/unchanged/orphaned chunks;
- minimum-work delta: embed+upsert only what changed, metadata-only updates where only title/ACL moved, deletes for orphans;
- the manifest is written **last**, after store writes, so a crash re-runs the same delta safely (idempotent retry) — the book's exact idempotency framing.

Supporting pieces also align:

- **Async ingestion** (Ch. 3 p. 74: decouple acknowledgement from background indexing): `app/documents.py` returns `202` and enqueues to an arq/Redis worker (`ingest/worker.py`). ✅
- **Cache invalidation on ingest** (Ch. 4 p. ~112: prefer event-driven purge over TTL-only): the worker calls `cache.invalidate_document(...)` after mutations — your semantic cache doesn't rely on TTL alone. ✅
- **Typed PII masking at ingestion** (Ch. 4 p. ~117): the book explicitly prefers entity-aware placeholders (`[DOCTOR_NAME] prescribed [MEDICATION]`) over `XXXX` nulling, because typed masks preserve answerability. `ingest/pii.py` replaces spans with `[EMAIL]`, `[SSN]`, `[PHONE]`, `[CREDIT_CARD]` — the book's preferred style, with overlap-resolution edge cases handled. ✅
- **Deterministic chunk ids** (`{doc_id}::{ordinal}`) make every downstream operation (upsert, delete, cache-purge, citation) stable — supports the book's entity-ID/version discipline (Ch. 6 failure-mode table).

---

## 4. Generation — grounded, cited, budgeted

| Book guidance | Implementation |
|---|---|
| Answer only from context; explicit refusal path (Ch. 4 p. ~107: *"If you don't know, say you don't know"*) | `SYSTEM_PROMPT` in `generation/prompts.py`: "You answer ONLY from the provided context… set refused=true… Do not guess." |
| Citations resolve to real sources (Ch. 3 UX: clickable lineage builds trust) | Numbered markers `[n]`, assembled by `core/context_assembly.py`, mapped back to chunk/doc ids into `Answer.citations` |
| Token-budget context assembly; TTFT grows superlinearly with context (Ch. 2 Table 2-3) | `assemble_context()` packs ranked chunks under `context_token_budget=4000` with tiktoken counting, dedup by chunk_id, tokenizer auto-selected per gen model |
| Cost-tiered models (Ch. 3: dynamic model routing) | 8B model for per-chunk contextual prefixes, 70B for answers, separate judge role — all routable via env |
| Structured output with defensive fallback | `GeneratedAnswer` pydantic schema; marker-scraping fallback if the model ignores the schema |

**Contextual chunk prefixes** (`ingest/contextual.py`) — prepending LLM-generated situational context to each chunk before embedding — is the Anthropic-style contextual retrieval technique the book treats as a production-grade enhancement, and you correctly run it with the cheap tier.

**Prompt-injection instruction defense** (Ch. 3 pp. 86–88): the book prescribes XML-style delimiting + explicit "context is data, never commands" instructions. Your prompt wraps context in literal `<context>…</context>` tags with exactly that rule, plus the system prompt repeats it. This is the book's defense #2 verbatim.

---

## 5. Guardrails — multilayered, fail-closed, leak-contained

The book's guardrail stack (Ch. 3 pp. 83–88, Ch. 4 pp. ~118–120): input sanitization, indirect-injection scanning of retrieved content, output checks (faithfulness/citations), and blocking disallowed content with logging.

Your implementation covers every layer:

- **Input:** `InjectionGuardrail` (heuristics + optional LLM escalation for borderline cases) and `PIIGuardrail` with redaction applied *before* the trace span is created (`core/pipeline.py` step 1) — raw PII never lands in observability.
- **Indirect injection:** retrieved chunk texts are scanned (`scan_for_injection`) and findings are attached to the trace *and* the answer metadata — detection without blocking, the right default.
- **Output:** `CitationGuardrail` (uncited claims rejected), `SchemaGuardrail`, output PII scan, and `GroundednessGuardrail` (LLM faithfulness check with wall-clock timeout, `fail_closed=False` so an evaluator outage degrades instead of blocking all traffic).
- **Fail-policy hygiene:** `GuardrailRunner._timed_check` guarantees a broken guard can never 500 a request; deterministic guards fail closed, probabilistic ones fail soft, and every exception is recorded with latency (`guardrails/runner.py`).
- **Block containment (SP2):** when an output guard blocks, the answer is replaced with a generic refusal and the block reason, payloads, and retrieved-id metadata are scrubbed from the returned object (reasons stay in traces only). This prevents leaking *why* to the caller — a detail most production systems get wrong.

---

## 6. Multi-tenancy & auth — identity from cryptography, not requests

- `app/api.py` derives tenant/tags **only** from a verified JWT (`require_principal`); the request body cannot express identity. The book's #1 tenancy foot-gun (client-supplied tenant ids) is structurally impossible here.
- Prod-mode validators reject unsafe combos (HS256 without secret, RS256 without JWKS, dev signer enabled).
- Tenant isolation is enforced at *every* store (vector, sparse, docstore, cache) and covered by dedicated tests (`test_multitenant_isolation.py`, `test_stores_acl.py`, `test_tenant_sparse_store.py`).
- Per-tenant sparse indexes (`TenantSparseStore`) mean one noisy tenant can't pollute another's lexical statistics.

---

## 7. Caching — the book's semantic-cache pattern, including its hardest problem

Ch. 4 (pp. ~111–113) recommends semantic caching with three hard problems: invalidation, eviction, scaling. Your `cache/` package:

- **Two tiers** — answer cache (bypasses the whole pipeline) and retrieval cache (bypasses both search legs) — matching the book's layered-cache table.
- **Embedding-keyed similarity lookup** with tenant + collection TAG scoping, so tenants can never serve each other's entries.
- **Event-driven invalidation**: doc-id tracking on every store + `invalidate_document` called from the ingest worker — the book's preferred answer to invalidation, which most implementations skip entirely in favor of TTL-only.
- Refused/blocked answers are never cached (`core/pipeline.py` store condition) — avoids poisoning the cache with refusals.

Qdrant itself is rated "Very High" for instant indexing in the book's Table 3-1 — a sound store choice for near-real-time freshness.

---

## 8. Evaluation — a real regression gate, statistically grounded

Ch. 6's core demand: evaluation must be a **CI gate** with a no-regression policy, not a notebook. Yours is:

- **Retrieval metrics** (`eval/retrieval_metrics.py`): precision/recall/F1@k, MRR, MAP, nDCG — the book's full classical suite (pp. 168–174).
- **Generation metrics** (`eval/generation_metrics.py`): groundedness/faithfulness, citation accuracy, answer relevancy — the book's "most critical generator metric" set.
- **LLM-as-judge with vote averaging** (`judge_votes=3`, fixed seed): directly implements the book's mitigation for judge stochasticity (*"run N times and average"*, p. 167).
- **Paired bootstrap regression gate** (`eval/gate.py` + `eval/stats.py`): aligns baseline/new runs item-by-item, computes the CI upper bound of the delta, fails on `ci_hi < −tolerance` or absolute floors, exits nonzero. This is statistically more rigorous than the book's own worked examples — paired bootstrap on aligned items controls for per-item difficulty variance.
- **Langfuse-native datasets/experiments** (`eval/experiment.py`, `dataset_cli.py`) give versioned benchmark data; a Ragas adapter exists for cross-checking.
- **Wired to CI**: `.github/workflows/eval-gate.yml` exists — the gate actually blocks merges, which the book argues is the difference between "having metrics" and "having evaluation."

---

## 9. Observability — per-stage tracing with privacy built in

- Tracing spans per stage (input guard, rewrite, retrieval, generation, output guard) with per-stage latencies, token usage, and **per-query cost** (`observability/cost.py`) — the book's system-metrics quartet (latency/tokens/cost/errors) at request granularity.
- Redaction-before-trace ordering (pipeline step 1) keeps raw PII out of spans; audit logs optionally salt-hash PII values (`pii_audit_value_hash` + mandatory salt validator).
- Self-hostable Langfuse v3/v4 stack in `infra/docker-compose.yml` — consistent with the book's guidance to keep sensitive telemetry inside your own boundary.

---

## Scorecard summary

| Book theme | Coverage |
|---|---|
| Two-flow architecture, clean layering | ✅✅ exemplary |
| Hybrid retrieval + RRF + cross-encoder rerank | ✅✅ exact match, incl. the book's model choice |
| ACL-scoped retrieval before fusion | ✅✅ stronger than most real systems |
| Incremental, idempotent, restartable ingestion | ✅✅ |
| Typed PII masking at ingest | ✅ |
| Grounded generation, citations, refusal path | ✅✅ |
| Injection defenses (direct + indirect) | ✅ |
| Multilayer guardrails with fail policies | ✅✅ |
| JWT-only identity, tenant isolation everywhere | ✅✅ |
| Semantic cache with event-driven invalidation | ✅ |
| Eval metrics + statistical CI gate in CI | ✅✅ |
| Per-stage observability with cost + PII hygiene | ✅ |

The remaining deltas — streaming, rate limiting, feedback capture, diversity reranking, richer parsing, latency hardening, judge independence — are catalogued with fixes in **Artifact 2** and prioritized in **Artifact 3**.
