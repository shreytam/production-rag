# SP4 · Retrieval Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hardens the retrieval wire-up path to ensure true hybrid and deterministic behaviour.

**Architecture:** We load the persisted BM25 sparse index from disk at the API/demo level via the `active_corpus` configuration knob and `PickleSparseIndexLoader` seam. When `version == 'full'`, we verify the loaded indexes and fail-closed with `HybridIndexError` if they are empty (preventing silent dense fallbacks). Chunker is updated to a single sliding window loop preserving overlap on continuation paragraph slices. RRF tie-breaks are sorted by `(-score, chunk_id)` for lexicographical stability. Context assembly tokenizer is parameterized with margin scaling, and rerankers share a normalized bounds candidate helper.

**Tech Stack:** Python 3.11-3.13, Pydantic, Tiktoken, Rank-BM25, Sentence-Transformers.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Empty sparse index fails closed by default under `version = 'full'`.
- Token assembly utilizes safety margin constraints mapping gen model to auto-resolved tiktoken encoders.

---

### Task 1: Configuration and Interface Hooks

**Files:**
- Modify: `core/config.py`
- Modify: `core/interfaces.py`

**Interfaces:**
- Consumes: None
- Produces:
  - Settings: `active_corpus`, `hybrid_require_sparse`, `sparse_index_dir`, `context_tokenizer`, `context_token_safety_margin`, `chunk_overlap`
  - `SparseIndexLoader` Protocol in `core/interfaces.py`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_config.py` checking settings loader hooks and `SparseIndexLoader` Protocol existence:
```python
import pytest
from pydantic import ValidationError
from core.config import Settings
from core.interfaces import SparseIndexLoader

def test_sp4_config_fields():
    settings = Settings(
        active_corpus="hotpotqa",
        hybrid_require_sparse=True,
        sparse_index_dir=".cache",
        context_tokenizer="auto",
        context_token_safety_margin=0.10,
        chunk_overlap=32
    )
    assert settings.active_corpus == "hotpotqa"
    assert settings.hybrid_require_sparse is True
    assert settings.context_tokenizer == "auto"

def test_sparse_index_loader_protocol():
    assert isinstance(SparseIndexLoader, type)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_config.py`
Expected: FAIL (Validation error on missing configuration fields)

- [ ] **Step 3: Modify files**
Add configuration fields to `Settings` inside `core/config.py` and register the `SparseIndexLoader` Protocol in `core/interfaces.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/config.py core/interfaces.py tests/test_sp4_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): register SP4 configuration parameters and sparse index loader interface" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Implement Sparse Index Loading

**Files:**
- Create: `providers/sparse/pickle_loader.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: `SparseIndexLoader` Protocol
- Produces: `PickleSparseIndexLoader` instance loading `bm25_{corpus}_{store}.pkl` mapped under `build_sparse_retriever(settings, corpus)`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_loader.py`:
```python
import pytest
from pathlib import Path
from core.registry import build_sparse_retriever
from providers.sparse.pickle_loader import PickleSparseIndexLoader
from providers.sparse.bm25 import BM25Retriever
from core.config import Settings

def test_pickle_loader_not_found(tmp_path):
    settings = Settings(sparse_index_dir=str(tmp_path))
    loader = PickleSparseIndexLoader(settings)
    assert loader.load("missing_corpus", "qdrant") is None

def test_build_sparse_retriever_falls_back_on_miss(tmp_path):
    settings = Settings(sparse_index_dir=str(tmp_path))
    retriever = build_sparse_retriever(settings, corpus="missing")
    assert isinstance(retriever, BM25Retriever)
    assert len(retriever._indices) == 0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_loader.py`
Expected: FAIL (Class `PickleSparseIndexLoader` does not exist and registry parameters mismatch)

- [ ] **Step 3: Modify files**
Create `providers/sparse/pickle_loader.py` implementing `PickleSparseIndexLoader` conforming to `SparseIndexLoader` Protocol. Apply D11 checks (check unpickles to `BM25Retriever` containing `_indices` dict with `len(_indices) >= 1`). Modify `core/registry.py` to route `build_sparse_retriever(settings, corpus=None)` to use the loader.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_loader.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add providers/sparse/pickle_loader.py core/registry.py tests/test_sp4_loader.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): implement PickleSparseIndexLoader and integrate with component factory" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Fail-Closed Retrieval Wire-up

**Files:**
- Modify: `core/pipeline.py`
- Modify: `app/api.py`
- Modify: `app/demo.py`
- Modify: `eval/run_eval.py`
- Modify: `eval/ragas_adapter.py`

**Interfaces:**
- Consumes: `build_sparse_retriever` from `core/registry`
- Produces: `HybridIndexError` on pipeline validation checks and parameterized `corpus` loading

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_pipeline_wire.py`:
```python
import pytest
from core.pipeline import build, HybridIndexError
from core.config import Settings

def test_pipeline_raises_on_empty_sparse_require_true(tmp_path):
    settings = Settings(
        sparse_index_dir=str(tmp_path),
        hybrid_require_sparse=True
    )
    with pytest.raises(HybridIndexError):
        build(version="full", corpus="missing", settings=settings)

def test_pipeline_warning_on_empty_sparse_require_false(tmp_path):
    settings = Settings(
        sparse_index_dir=str(tmp_path),
        hybrid_require_sparse=False
    )
    pipeline = build(version="full", corpus="missing", settings=settings)
    assert pipeline is not None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_pipeline_wire.py`
Expected: FAIL (No HybridIndexError and missing validations in build)

- [ ] **Step 3: Modify files**
Modify `core/pipeline.py` to add validation, `HybridIndexError`, and parameter mapping supporting backward-compatible `dataset kwarg` overrides. Modify `app/api.py`, `app/demo.py`, `eval/run_eval.py`, and `eval/ragas_adapter.py` call sites to pass configuration corpus attributes.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_pipeline_wire.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/pipeline.py app/api.py app/demo.py eval/run_eval.py eval/ragas_adapter.py tests/test_sp4_pipeline_wire.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): wire fail-closed sparse index check and update API and eval build calls" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Chunker Token Conservation

**Files:**
- Modify: `ingest/chunking.py`

**Interfaces:**
- Consumes: tiktoken encoder
- Produces: `chunk_document` output retaining overlaps with single-pass implementation

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_chunker_conservation.py`:
```python
import pytest
from core.types import Document
from ingest.chunking import chunk_document

def test_chunker_retains_all_tokens_oversized():
    # Construct a paragraph longer than max_tokens=10 with overlap=2
    para = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    doc = Document(doc_id="d1", text=para, tenant_id="t1")
    
    chunks = chunk_document(doc, max_tokens=10, overlap=2)
    # Check no words are truncated across slices
    combined_words = " ".join([c.text for c in chunks])
    for word in para.split():
         assert word in combined_words
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_chunker_conservation.py`
Expected: FAIL (Trim drops trailing paragraph continuation tokens)

- [ ] **Step 3: Modify files**
Scrub `_token_chunks_from_paragraph` to sliding-window chunks with overlap. Rewrite `chunk_document` to use a single unified packing loop preserving trailing boundaries during window slicing, and clean out the duplicate execution block.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_chunker_conservation.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add ingest/chunking.py tests/test_sp4_chunker_conservation.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): rewrite chunk_document to prevent token loss on oversized splits and preserve overlaps" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: RRF Tie-Break Determinism

**Files:**
- Modify: `core/rrf.py`

**Interfaces:**
- Consumes: ScoredChunk lists
- Produces: Lexicographically stable output rankings

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_rrf_ties.py`:
```python
import pytest
from core.types import Chunk, ScoredChunk, RetrievalSource
from core.rrf import reciprocal_rank_fusion

def test_rrf_stable_tie_breaks():
    c1 = Chunk(chunk_id="ch_a", doc_id="d1", text="text1", tenant_id="t1")
    c2 = Chunk(chunk_id="ch_b", doc_id="d1", text="text2", tenant_id="t1")
    
    sc1 = ScoredChunk(chunk=c1, score=1.0, source=RetrievalSource.DENSE)
    sc2 = ScoredChunk(chunk=c2, score=1.0, source=RetrievalSource.DENSE)
    
    # Shuffle rankings to simulate order-dependence hazards
    ranking1 = [sc1, sc2]
    ranking2 = [sc2, sc1]
    
    r1 = reciprocal_rank_fusion([ranking1], k=60)
    r2 = reciprocal_rank_fusion([ranking2], k=60)
    
    # Outputs must have exactly identical ordering despite different raw orders
    assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]
    assert r1[0].chunk_id == "ch_a"
    assert r1[1].chunk_id == "ch_b"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_rrf_ties.py`
Expected: FAIL (Sorting is Dict order-dependent and fails on shuffled inputs)

- [ ] **Step 3: Modify files**
Update `core/rrf.py` to sort tie-breaks lexicographically: `ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_rrf_ties.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/rrf.py tests/test_sp4_rrf_ties.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): enforce stable lexicographical tie breaking on RRF fusion lists" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: Context Assembly Tokenizer Alignment

**Files:**
- Modify: `core/context_assembly.py`
- Modify: `generation/grounded_generator.py`

**Interfaces:**
- Consumes: Config settings `context_tokenizer` and `context_token_safety_margin`
- Produces: Alignment of tokens to Llama-like settings and safety scaling margins

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_tokenizer_budget.py`:
```python
import pytest
from core.context_assembly import resolve_encoding
from core.config import Settings

def test_resolve_tokenizer_encoding():
    assert resolve_encoding("cl100k_base", "llama-model") == "cl100k_base"
    assert resolve_encoding("auto", "gpt-4o") == "o200k_base"
    assert resolve_encoding("auto", "llama-3-model") == "cl100k_base"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_tokenizer_budget.py`
Expected: FAIL (Missing resolve_encoding and non-parameterized count_tokens)

- [ ] **Step 3: Modify files**
Implement `resolve_encoding` inside `core/context_assembly.py`. Modify functions `count_tokens` and `assemble_context` to accept `encoding_name`. Hook `GroundedGenerator.__init__` and `GroundedGenerator.generate` to query resolved encoders and scale budgets by safety margin.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_tokenizer_budget.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/context_assembly.py generation/grounded_generator.py tests/test_sp4_tokenizer_budget.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): parameterize budgeting tokenizer and apply safety margin checks" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: Reranker Normalization Helper

**Files:**
- Create: `providers/rerankers/_common.py`
- Modify: `providers/rerankers/local_cross_encoder.py`
- Modify: `providers/rerankers/nim_rerank.py`

**Interfaces:**
- Consumes: Candidate chunks and indexes
- Produces: `normalize_candidates` returning clean ranks safely

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp4_reranker_common.py`:
```python
import pytest
from core.types import Chunk, ScoredChunk, RetrievalSource
from providers.rerankers._common import normalize_candidates

def test_normalize_candidates_drops_bounds_failures():
    c1 = Chunk(chunk_id="c_1", doc_id="d1", text="a", tenant_id="t")
    sc1 = ScoredChunk(chunk=c1, score=0.5)
    
    # Index 2 is out of range (only len=1)
    scored = [(2, 0.9), (0, 0.8)]
    clean = normalize_candidates([sc1], scored, top_n=2)
    
    assert len(clean) == 1
    assert clean[0].chunk_id == "c_1"
    assert clean[0].score == 0.8
    assert clean[0].source == RetrievalSource.RERANK
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp4_reranker_common.py`
Expected: FAIL (No `providers/rerankers/_common.py` module exists)

- [ ] **Step 3: Modify files**
Create `providers/rerankers/_common.py` implementing `normalize_candidates`. Update `rerank` methods in both `providers/rerankers/local_cross_encoder.py` and `providers/rerankers/nim_rerank.py` to route candidate mappings through the helper.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp4_reranker_common.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add providers/rerankers/_common.py providers/rerankers/local_cross_encoder.py providers/rerankers/nim_rerank.py tests/test_sp4_reranker_common.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(correctness): deploy common normalize_candidates reranker helper to standardise scoring outcomes" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
