# SP12 · Native Query Rewriting & Synonym Expansion — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 2. Introduces a modular, hybrid query rewriting component to improve retriever recall by mapping business-domain synonyms and translating shorthand inputs prior to search. Connects between the input-guardrail and cache check stages in `RAGPipeline.answer()`.

---

## 1. Context & problem

The RAG search pipeline executes user queries directly as typed (`retrieval/hybrid.py`). If a query uses an acronym (e.g. "NYPD"), an internal product shorthand (e.g. "Jupiter" instead of "Jupiter-Next"), or a brief phrase (e.g. "sales reports"), search quality suffers:
- **BM25 (Sparse)** fails due to keyword mismatch: an index containing "New York Police Department" fails to match "NYPD".
- **Dense/Embedding (Dense)** searches degrade: narrow domain terms may not reside closely in the model's vocabulary space.

Currently:
- There is **no query expansion or translation** seam. Every search uses raw text.
- Standard RAG solutions run a heavy LLM query rewriting call on *every* request, which adds 300–600ms latency and scales costs unnecessarily.
- Tenant-specific terminology and slang differ: "project A" in Tenant 1 may mean "Project Orion" in Tenant 2. A shared, static dictionary is invalid and violates tenant isolation.

---

## 2. Goals

- **G1 — Hybrid Rewriting Path.** Implement a two-tiered rewriter: a deterministic helper executing tenant-specific synonym replacements, falling back to a structured LLM query-expansion call for complex queries.
- **G2 — Hostile Tenant Isolation.** Tenant synonym dictionary mappings are stored securely in Redis under tenant-specific namespaces (`rewriter:synonyms:{tenant_id}`) to prevent cross-tenant leakage.
- **G3 — Performance-safe placement.** Rewriting executes *before* the L1/L2 caches in the query pipeline. Rewritten inputs are stored in cache keys to ensure rewriting costs are bypassed on cache hits.
- **G4 — CI Measurability.** Rewriting is enabled in evaluation runs by default, showing direct impacts on SQuAD/HotpotQA recall bounds.

---

## 3. Non-goals (deferred)

- **Interactive query clarification** → out of scope; query rewriting occurs silently on the backend, without asking the user for follow-up inputs.
- **Self-learning synonyms** → out of scope; synonym lists are loaded administratively into Redis rather than inferred dynamically from user clicks.

---

## 4. Decisions (PROPOSED)

| # | Decision | Choice (proposed) | Rationale |
|---|---|---|---|
| D1 | Rewriting Strategy | **Hybrid Rules + LLM** | Combines low-latency deterministic dictionary lookups with fallback semantic flexibility. |
| D2 | Synonym Storage | **Redis Store** | Allows real-time dictionary updates per tenant without restarting servers. |
| D3 | Cache Interaction | **Before Caching** | Caches the final generated query execution, avoiding repeat LLM rewriting costs. |
| D4 | Eval Gate Integration | **Enabled in CI** | Ensures SQuAD and HotpotQA adapter runs verify the impact of rewritten terms. |
| D5 | Fallback Criteria | **Triggers on complex sentences / mismatch** | Simple keyword queries hit the quick synonym path; long questions trigger query expansion. |

---

## 5. Architecture & components

```
RAGPipeline.answer(question, acl)
   │
   ▼
[Input Guardrails] (redacts PII/injection)
   │
   ▼
[Query Rewriter] 
   ├─ 1. Query Redis for hash: rewriter:synonyms:{acl.tenant_id}
   ├─ 2. Apply regex replace on matched key boundaries
   └─ 3. (If token length > threshold or no rule matches) -> Complete LLM query expansion
   │
   ▼
[L1/L2 Cache Check] (exact/semantic check on rewritten_question)
   ├─ Hit  ─► Return cached answer
   └─ Miss ─► [Retriever] ─► [Generator] ─► [Output Guard] ─► [Cache Store] ──► Return
```

### 5.1 `QueryRewriter` Protocol — `core/interfaces.py`
```python
class QueryRewriter(Protocol):
    def rewrite(self, query: str, acl: ACLContext) -> str: ...
```

### 5.2 `HybridQueryRewriter` — `providers/rewriter/hybrid_rewriter.py`
- Consumes a `redis` connection client and a cheap `Generator` instance (`role="context"` or dedicated `role="rewriter"`).
- Reads `rewriter:synonyms:{acl.tenant_id}` hash maps from Redis.
- Performs regex word-boundary synonym replacements (case-insensitive).
- If the query exceeds a configured length threshold (e.g. `rewriter_llm_threshold=5` words) and no synonym was matching, it passes the query to a prompt template:
  ```
  You are an expert search assistant. Rewrite this RAG query to maximize search retrieval matching.
  Generate a single descriptive paragraph. Keep specialized terms and expand acronyms.
  Query: {query}
  ```
- Returns the expanded/rewritten query.
- Gracefully degrades: if Redis is unreachable or the LLM fails, returns the raw `query` unchanged (fail-soft).

---

## 6. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `rewriter_enabled` | `True` | Master switch for the query rewriter stage. |
| `rewriter_llm_enabled` | `True` | Enables the fallback LLM-based query-expansion step. |
| `rewriter_llm_threshold` | `5` | Word count threshold above which complex queries trigger LLM expansion. |
| `rewriter_redis_url` | `"redis://localhost:6379/1"` | Redis database url for synonym storage dictionary lookup. |

---

## 7. Testing & Parity (TDD)

1. **`test_synonym_lookup`**: Verify that if a tenant has `"NYPD" -> "New York Police Department"` key-value mapped in Redis, queries containing `"NYPD"` are replaced appropriately.
2. **`test_tenant_synonym_isolation`**: Verify that tenant B's synonym mappings never affect tenant A's query rewrites.
3. **`test_llm_expansion_trigger`**: Verify that queries under the word threshold skip the LLM call, while longer queries execute the generator completion.
4. **`test_pipeline_integration`**: Verify that mutated questions are routed to retrieval and cached correctly.

---

## 8. Files

**Create**
- `providers/rewriter/__init__.py`
- `providers/rewriter/hybrid_rewriter.py`
- `tests/test_query_rewriter.py`

**Modify**
- `core/interfaces.py` — add `QueryRewriter` Protocol.
- `core/config.py` — register configuration knobs.
- `core/registry.py` — add `build_query_rewriter`.
- `core/pipeline.py` — wire rewriter inside `answer`.
- `pyproject.toml` — verify `redis` client availability.
