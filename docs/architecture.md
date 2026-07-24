# Architecture Deep-Dive

This document describes the component design, data-flow, and interface
contracts for the Production RAG system. The README has the quickstart and
high-level diagrams; this file has the details you need to extend or debug
a specific layer.

---

## Component Map

```
core/                   ← contracts + wire types + config (Wave 0; stable)
  interfaces.py         ← Protocol definitions
  types.py              ← Pydantic wire models
  config.py             ← Settings (pydantic-settings, env-driven)
  registry.py           ← Concrete class selection (the only place)
  pipeline.py           ← RAGPipeline.build(version, dataset)
  rrf.py                ← Reciprocal Rank Fusion
  context_assembly.py   ← Token-budgeted context packing

providers/              ← Swappable implementations of core.interfaces
  embedders/            ← OpenAICompatibleEmbedder (NIM or OpenAI)
  generators/           ← OpenAICompatibleGenerator, AnthropicGenerator
  rerankers/            ← LocalCrossEncoderReranker (BGE), NIMReranker
  sparse/               ← BM25Retriever (rank-bm25)
  vectorstores/         ← QdrantStore, PgvectorStore

ingest/
  run.py                ← CLI entry point: loads → chunks → embeds → upserts + BM25 index
  chunking.py           ← Fixed-size sliding-window chunker; produces Chunk objects
  contextual.py         ← ContextualPrefixer: LLM call per chunk, disk-cached
  pii.py                ← PIIRedactor: regex-based; runs before chunking

corpora/                ← Dataset adapters (one package per corpus)
  hotpotqa/adapter.py   ← HotpotQAAdapter: load(limit) → list[Document]
                                           build_golden(limit) → list[GoldenItem]
  arxiv/adapter.py
  financebench/adapter.py

retrieval/
  hybrid.py             ← DenseRetriever, HybridRetriever

generation/
  grounded_generator.py ← GroundedGenerator: context assembly + structured output
  prompts.py            ← System prompt + build_user_prompt()

guardrails/
  input_injection.py    ← InjectionGuardrail: heuristics + optional LLM second-opinion

eval/
  experiment.py         ← Langfuse-native runner: dataset items → pipeline.run → evaluators → Langfuse dataset run (SDK run_experiment)
  gate.py               ← reads run scores back from Langfuse; paired-bootstrap and/or thresholds; exits 1 on regression
  evaluators.py         ← wraps the metric fns below as Langfuse evaluators
  langfuse_eval.py      ← EvalBackend protocol + normalized types (the SDK seam)
  _langfuse_backend.py  ← real Langfuse SDK backend (lazy-imported; only module importing langfuse in eval/)
  dataset_cli.py        ← seed a dataset from a local file; add-from-trace to promote a prod trace
  generation_metrics.py ← faithfulness, answer_relevancy, context_precision, context_recall
  retrieval_metrics.py  ← recall_at_k, precision_at_k, ndcg_at_k, mrr
  llm_judge.py          ← Holistic LLM judge (single score 0–1)
  stats.py              ← bootstrap_ci, paired_bootstrap (1 000 resamples)
  fast_subset.py        ← Stratified 15-item subset for CI --fast

observability/
  langfuse_tracing.py   ← Per-query trace (scores, tokens, latency, cost)
  cost.py               ← Token-to-USD cost estimation

app/
  api.py                ← FastAPI: POST /query → Answer JSON
  demo.py               ← Streamlit interactive demo
```

---

## Interface Contracts (`core/interfaces.py`)

All pipeline components depend only on these Protocols. Concrete classes are
selected in `core/registry.py` based on `Settings`. Adding a new provider
means implementing the Protocol and adding a branch to registry — no changes
to pipeline code.

### `Embedder`

```python
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...
```

### `VectorStore`

```python
class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None: ...
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def search(self, embedding: Vector, top_k: int, acl: ACLContext) -> list[ScoredChunk]: ...
    def count(self, acl: ACLContext | None = None) -> int: ...
```

`acl` is a **required** argument to `search`. Both `QdrantStore` and
`PgvectorStore` apply it as a **pre-similarity filter condition**, not a
post-hoc Python filter. This is the security boundary.

### `SparseRetriever`

```python
class SparseRetriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, top_k: int, acl: ACLContext) -> list[ScoredChunk]: ...
```

### `Reranker`

```python
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...
```

### `Generator`

```python
class Generator(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
```

When `response_model` is provided, implementations must populate
`LLMResponse.parsed` as a dict matching that model's fields.

### `Guardrail`

```python
class Guardrail(Protocol):
    @property
    def name(self) -> str: ...
    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult: ...
```

---

## Data-Flow: Query Path

```
User request (question: str, caller_acl: ACLContext)
    │
    ▼
InjectionGuardrail.check(question)
    │  BLOCK → return error immediately
    ▼ PASS
RAGPipeline.run(question, acl)
    │
    ├─ DenseRetriever.retrieve(Query)
    │     │  Embedder.embed_query(question) → Vector
    │     │  VectorStore.search(vector, top_k=20, acl) → list[ScoredChunk]
    │     └─ returns list[ScoredChunk]
    │
    ├─ SparseRetriever.search(question, top_k=20, acl) → list[ScoredChunk]  [full only]
    │
    ├─ rrf.fuse([dense_hits, sparse_hits], k=60) → list[ScoredChunk]        [full only]
    │
    ├─ Reranker.rerank(question, fused, top_n=8) → list[ScoredChunk]        [full only]
    │
    ├─ context_assembly.assemble_context(chunks, token_budget=4000)
    │     → ContextWindow { text: str, items: list[(int, ScoredChunk)], marker_to_chunk }
    │
    └─ GroundedGenerator.generate(question, chunks)
          │  Generator.complete([system, user], response_model=GeneratedAnswer)
          │  reconcile citation markers → list[Citation]
          └─ returns Answer { text, citations, contexts, usage, refused }
```

---

## Semantic Cache (`cache/`)

Opt-in (`cache_enabled = False` by default; requires Redis 8). When enabled,
`RAGPipeline.build()` wires **two** cache tiers, each the same `SemanticCache`
Protocol implementation bound to its own redis-vl index:

- **Answer tier** (`rag_cache_answer`) — `query → final Answer`. A hit skips
  retrieval **and** generation (and output guardrails — the stored answer was
  already vetted when it was cached).
- **Retrieval tier** (`rag_cache_retrieval`) — `query → reranked chunk set`. A
  hit skips retrieval only; generation always runs fresh.

Both tiers key on **semantic** similarity — a cosine-distance vector search on
the embedded (redacted) query, not exact string match — so paraphrased repeat
questions still hit. `core/config.py` knobs: `cache_enabled`,
`cache_similarity_threshold` (default `0.9`), `cache_ttl_seconds` (default
`3600`).

**Backend:** `cache/_redisvl_backend.py` (`RedisVLSemanticCache`) is the only
module that imports `redisvl`, and only lazily inside method bodies — the
offline test suite and lint never need Redis or the package installed. It
targets **Redis 8**, which folds the vector/search engine into core (no
separate Redis Stack image). `tests/cache/fake_cache.py` (`FakeSemanticCache`)
is an in-memory equivalent used by the entire offline suite.

**Isolation:** every entry is tagged `tenant_id` + `collection_id` (mandatory
TAG filters on both store and lookup) — a cache hit can never cross tenants or
collections, mirroring the ACL security model above.

**Invalidation:** `ingest/worker.py`'s `run_ingest`/`run_delete`, after
committing to Qdrant, call `invalidate_document(tenant_id, collection_id,
doc_id)` on both tiers — a filtered delete over the `doc_ids` TAG reverse
index, so every cached entry citing a changed/deleted document is evicted
precisely (no dangling citations). A brand-new document has no id to match
against, so a per-entry **TTL** (`cache_ttl_seconds`) backstops that
new-document blind spot until the entry self-expires.

**Eval bypass:** `eval/experiment.py` builds the pipeline via `pipeline.build`
with the cache off, exactly like guardrails are forced off — cache hits can
never confound Langfuse metrics. Refusals and guardrail-blocked answers are
never written to the cache.

---

## Data-Flow: Ingest Path

```
ingest.run --dataset hotpotqa --limit N
    │
    ▼
HotpotQAAdapter.load(limit)          → list[Document]
    │
    ▼
PIIRedactor.redact(doc.text)         → redacted text (pre-chunking)
    │
    ▼
chunk_document(doc)                  → list[Chunk]  (fixed-size, overlapping)
    │
    ├─ [--contextual] ContextualPrefixer.annotate(chunks, doc_texts)
    │     → Chunk.contextual_prefix set for each chunk
    │     (one Generator.complete call per chunk, disk-cached in .cache/contextual/)
    │
    ▼
Embedder.embed_documents([c.embed_text for c in chunks])  → list[Vector]
    │   (embed_text = prefix + "\n\n" + text  if prefix else text)
    │
    ├─ VectorStore.ensure_collection(dim) + VectorStore.upsert(chunks_with_embeddings)
    │
    ├─ SparseRetriever.index(chunks)  → BM25 index pickled to .cache/bm25_{ds}_{store}.pkl
    │
    └─ HotpotQAAdapter.build_golden(limit)
          → list[GoldenItem { question, answer, relevant_doc_ids, tenant_id }]
          written to data/eval/hotpotqa.json
```

---

## Data-Flow: Eval Path

Evaluation is a **Langfuse experiment**: the dataset and every run live on the
hosted Langfuse; the loop is driven by the SDK's `run_experiment`. The existing
metric functions are reused verbatim, re-wrapped as Langfuse evaluators.

```
eval.experiment --dataset hotpotqa --version full --fast --run-name <name>
    │
    ▼
backend.get_dataset_items("hotpotqa")   → Langfuse dataset items (input/expected_output/metadata)
fast_subset(items, n=15)                → 15 stratified items
    │
    run_experiment(name=hotpotqa, run_name=<name>, data=items, task, evaluators):
        task:  pipeline.run(question, acl)   → { answer, retrieved_ids, contexts, … }   (linked trace per item)
        evaluators (client-side, pushed as Langfuse scores):
            retrieval  → recall_at_5, precision_at_5, ndcg_at_5, mrr
            generation → faithfulness, answer_relevancy, context_precision, context_recall
            judge      → judge_score
    │
    a named Langfuse dataset run, per-item traces + scores (compare runs in the Langfuse UI)
```

```
eval.gate --dataset hotpotqa --new-run <name> --baseline-run baseline
    │
    backend.get_run_scores(<name>)     → per-item scores (via get_dataset_run + api.scores.get_many)
    backend.get_run_scores(baseline)
    align by dataset-item id
    │
    per metric, per eval_gate_mode (bootstrap | threshold | both):
        bootstrap:  paired_bootstrap(base_items, new_items) → (_, lo, hi); FAIL if hi < −tolerance
        threshold:  FAIL if new_mean < floor (eval_gate_thresholds)
    │
    print table; exit 0 (PASS) or exit 1 (FAIL)
```

---

## ACL Security Model

`ACLContext` has two fields: `tenant_id` (required) and `acl_tags` (optional
tuple). The ACL rule (`ACLContext.allows`):

1. Tenant must match exactly.
2. A chunk with no `acl_tags` is visible to any caller in the same tenant.
3. A chunk with `acl_tags` requires the caller to hold at least one matching
   tag (capability-style).

This is enforced at the store level: `VectorStore.search` and
`SparseRetriever.search` receive the caller's `ACLContext` and apply it as
a filter **before** computing similarity scores. There is no code path that
returns un-ACL-filtered results.

**Never derive ACLContext from the prompt or retrieved text.** It must come
from the authenticated caller context (session, JWT, etc.).

---

## Pipeline Versions

`pipeline.build(version, dataset)` produces one of two retrieval strategies:

| Version | Retriever | Components used |
|---|---|---|
| `baseline` | `DenseRetriever` | Embedder + VectorStore only |
| `full` | `HybridRetriever` | Embedder + VectorStore + SparseRetriever + RRF + Reranker |

Both use the same `GroundedGenerator`. The baseline exists to measure the
contribution of each component layer.

---

## Bootstrapping the Baseline Run (CI prerequisite)

The gate compares each PR's run against a Langfuse dataset run named `baseline`
for the `hotpotqa` dataset. That baseline run must exist on the hosted Langfuse
before the CI gate can pass. The CI `eval` job requires the `LANGFUSE_HOST`,
`LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` secrets (plus `NVIDIA_API_KEY`).

To bootstrap it:

```bash
# With the stack running, NVIDIA_API_KEY and LANGFUSE_* set:
uv run python -m ingest.run --dataset hotpotqa --limit 50   # writes data/eval/hotpotqa.json
make seed DATASET=hotpotqa ITEMS=data/eval/hotpotqa.json    # upload golden items to Langfuse
uv run python -m eval.experiment --dataset hotpotqa --version full --fast --run-name baseline
```

The baseline run **must** use the same `--fast` (15-item subset) that the CI
workflow uses, or the comparison will be apples-to-oranges. Dataset items are
curated in the Langfuse UI (or promoted from traces via
`python -m eval.dataset_cli add-from-trace`).

---

## LLM Call Budget per CI Run

Estimated for `--limit 50 --fast` (15 items after `fast_subset`):

| Stage | Calls per item | Total (15 items) |
|---|---|---|
| Dense embed (query) | 1 embed | 15 embed |
| Generation (answer) | 1 gen | 15 gen |
| Faithfulness | 2 gen | 30 gen |
| Answer Relevancy | 1 gen + 3 embed | 15 gen + 45 embed |
| Context Precision | 1 gen × 8 contexts | 120 gen |
| Context Recall | 2 gen | 30 gen |
| Holistic Judge | 1 gen | 15 gen |
| **Total** | | **~225 gen + ~60 embed** |

At NIM free-tier (~40 rpm), wall time is approximately 8–12 minutes.
Cost on NIM free tier: effectively zero (quota-limited, not billed per token).
