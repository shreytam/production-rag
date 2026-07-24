# Redis Semantic Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-tier, tenant-isolated semantic cache (answer + retrieval) backed by Redis 8 / redis-vl that serves semantically-equivalent repeat queries while never returning stale or cross-tenant results.

**Architecture:** A `cache/` subsystem exposes a `SemanticCache` Protocol instantiated twice (answer tier, retrieval tier), each a redis-vl vector index. `RAGPipeline.answer` consults the answer tier (skip retrieval+generation on hit), then the retrieval tier (skip retrieval on hit). Ingest/delete workers call `invalidate_document` (RediSearch TAG-filtered delete = reverse index) for precise eviction; a per-entry TTL backstops the new-document blind spot.

**Tech Stack:** Python 3.12, Pydantic v2, redis-vl on Redis 8 (query/vector engine in core), pytest. Offline suite runs against an in-memory `FakeSemanticCache`; the real backend is exercised by one opt-in live smoke test.

## Global Constraints

- **Commit authorship:** every commit authored SOLELY as `Shreytam Goyal <shreytamgoyal@gmail.com>`. NO Claude/AI attribution of any kind (no `Co-Authored-By`, no `Claude-Session`, no "Generated with"). Do NOT use the codiant identity. Commit form: `git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "..."`.
- **Import isolation:** NO top-level `import redis_vl` / `redisvl` anywhere in `cache/`. Only `cache/_redisvl_backend.py` may import it, and ONLY lazily inside method bodies (including `__init__`). Importing any `cache/` module — and thus lint + the offline suite — must need neither Redis nor the `redis-vl` package.
- **Tenant/collection isolation:** every cache lookup/store/eviction is scoped by `tenant_id` AND `collection_id`. A hit must never cross either.
- **Never cache refusals or guardrail-blocked answers.**
- **Eval path never uses the cache.**
- **Default off:** `cache_enabled=False`; the subsystem is inert unless explicitly enabled.
- **Test runner:** `.venv/bin/python -m pytest <path> -v` (NOT `uv run`). Exit code 0 is the signal.
- `collection_id` is `str | None`; normalize `None` to the sentinel string `"__none__"` for TAG scoping — do this consistently in every tier and the fake.

---

## File Structure

**New**
- `cache/__init__.py` — package marker.
- `cache/semantic_cache.py` — `SemanticCache` Protocol, payload (de)serialization helpers, `build_cache()`.
- `cache/_redisvl_backend.py` — `RedisVLSemanticCache` (the only redis-vl importer, lazy).
- `tests/cache/__init__.py`, `tests/cache/fake_cache.py` — `FakeSemanticCache`.
- `tests/test_cache_semantics.py` — fake-backed contract tests (threshold, isolation, eviction, TTL).
- `tests/test_cache_pipeline.py` — pipeline cache behavior.
- `tests/test_cache_worker.py` — worker invalidation.
- `tests/test_cache_backend.py` — import-isolation + `build_cache` construction.
- `tests/test_cache_live_smoke.py` — opt-in live redis-vl round-trip.

**Modified**
- `core/config.py` — three cache knobs.
- `core/pipeline.py` — cache-aware `answer()`; `build()` wires cache + embedder.
- `ingest/worker.py` — `IngestDeps.caches`; `run_ingest`/`run_delete` invalidate after commit.
- `infra/docker-compose.yml` — `redis:7` → `redis:8`.
- `pyproject.toml` / `uv.lock` — `redis-vl` extra.
- `.env.example` — cache vars.
- `docs/architecture.md`, `docs/PROJECT_STATUS.md` — cache section.

---

### Task 1: Config knobs

**Files:**
- Modify: `core/config.py` (add near the `redis_url` block, ~line 181)
- Test: `tests/test_cache_config.py`

**Interfaces:**
- Produces: `Settings.cache_enabled: bool` (default `False`), `Settings.cache_similarity_threshold: float` (default `0.9`), `Settings.cache_ttl_seconds: int` (default `3600`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_config.py
from core.config import Settings


def test_cache_defaults_are_conservative():
    s = Settings()
    assert s.cache_enabled is False
    assert s.cache_similarity_threshold == 0.9
    assert s.cache_ttl_seconds == 3600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'cache_enabled'`.

- [ ] **Step 3: Add the knobs**

In `core/config.py`, directly after the `ingest_queue_name` line in the async-ingest block:

```python
    # --- Semantic cache (Redis 8 / redis-vl; opt-in) ---
    cache_enabled: bool = False
    cache_similarity_threshold: float = 0.9
    cache_ttl_seconds: int = 3600
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cache_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_cache_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(cache): add opt-in semantic cache config knobs"
```

---

### Task 2: `SemanticCache` Protocol + serialization + `build_cache`

**Files:**
- Create: `cache/__init__.py` (empty)
- Create: `cache/semantic_cache.py`
- Test: `tests/test_cache_serialization.py`

**Interfaces:**
- Consumes: `core.types.Answer`, `core.types.ScoredChunk`; `core.config.Settings`.
- Produces:
  - `class SemanticCache(Protocol)` with `lookup(*, tenant_id: str, collection_id: str | None, embedding: Sequence[float]) -> dict | None`, `store(*, tenant_id: str, collection_id: str | None, embedding: Sequence[float], payload: dict, doc_ids: Sequence[str]) -> None`, `invalidate_document(*, tenant_id: str, collection_id: str | None, doc_id: str) -> int`.
  - `COLLECTION_NONE = "__none__"` and `norm_collection(collection_id: str | None) -> str`.
  - `answer_to_payload(ans: Answer) -> dict`, `answer_from_payload(payload: dict) -> Answer`.
  - `scored_to_payload(scored: list[ScoredChunk]) -> dict`, `scored_from_payload(payload: dict) -> list[ScoredChunk]`.
  - `doc_ids_of(scored: list[ScoredChunk]) -> list[str]` (deduped, order-preserving).
  - `build_cache(settings: Settings) -> tuple[SemanticCache, SemanticCache]` (answer tier, retrieval tier) — pure constructor, lazily imports the backend.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_serialization.py
from cache.semantic_cache import (
    COLLECTION_NONE, answer_from_payload, answer_to_payload, doc_ids_of,
    norm_collection, scored_from_payload, scored_to_payload,
)
from core.types import Answer, Chunk, Citation, ScoredChunk


def _scored():
    return [
        ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="t1", tenant_id="acme"), score=0.9),
        ScoredChunk(chunk=Chunk(chunk_id="c2", doc_id="d1", text="t2", tenant_id="acme"), score=0.8),
        ScoredChunk(chunk=Chunk(chunk_id="c3", doc_id="d2", text="t3", tenant_id="acme"), score=0.7),
    ]


def test_norm_collection():
    assert norm_collection(None) == COLLECTION_NONE
    assert norm_collection("kb") == "kb"


def test_answer_round_trip():
    ans = Answer(text="hi", citations=[Citation(marker=1, chunk_id="c1")], refused=False)
    back = answer_from_payload(answer_to_payload(ans))
    assert back.text == "hi"
    assert back.refused is False
    assert back.citations[0].chunk_id == "c1"


def test_scored_round_trip_and_doc_ids():
    sc = _scored()
    back = scored_from_payload(scored_to_payload(sc))
    assert [s.chunk_id for s in back] == ["c1", "c2", "c3"]
    assert doc_ids_of(sc) == ["d1", "d2"]  # deduped, order-preserving
```

> Note: confirm `Citation`'s field names by reading `core/types.py` before writing the test; adjust `Citation(...)` kwargs to match (the round-trip assertion is what matters, not the exact constructor args).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_serialization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cache'`.

- [ ] **Step 3: Create the package and module**

`cache/__init__.py`: empty file.

`cache/semantic_cache.py`:

```python
"""Semantic cache seam: a tenant-scoped, embedding-keyed cache Protocol plus the
payload (de)serialization shared by the answer and retrieval tiers.

NO top-level redis-vl import lives here or anywhere else in this package except
_redisvl_backend.py (lazily). Importing this module needs neither Redis nor the
redis-vl package, so lint and the offline suite stay infra-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from core.types import Answer, ScoredChunk

if TYPE_CHECKING:
    from core.config import Settings

COLLECTION_NONE = "__none__"


def norm_collection(collection_id: str | None) -> str:
    """Map an absent collection to a stable sentinel so TAG scoping is total."""
    return collection_id if collection_id else COLLECTION_NONE


@runtime_checkable
class SemanticCache(Protocol):
    def lookup(self, *, tenant_id: str, collection_id: str | None,
               embedding: Sequence[float]) -> dict | None: ...

    def store(self, *, tenant_id: str, collection_id: str | None,
              embedding: Sequence[float], payload: dict,
              doc_ids: Sequence[str]) -> None: ...

    def invalidate_document(self, *, tenant_id: str, collection_id: str | None,
                            doc_id: str) -> int: ...


def answer_to_payload(ans: Answer) -> dict:
    return {"kind": "answer", "answer": ans.model_dump(mode="json")}


def answer_from_payload(payload: dict) -> Answer:
    return Answer.model_validate(payload["answer"])


def scored_to_payload(scored: list[ScoredChunk]) -> dict:
    return {"kind": "scored", "scored": [s.model_dump(mode="json") for s in scored]}


def scored_from_payload(payload: dict) -> list[ScoredChunk]:
    return [ScoredChunk.model_validate(s) for s in payload["scored"]]


def doc_ids_of(scored: list[ScoredChunk]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in scored:
        d = s.chunk.doc_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_cache(settings: "Settings") -> tuple[SemanticCache, SemanticCache]:
    """Construct the (answer, retrieval) tier pair. Pure constructor — does NOT
    consult cache_enabled (the caller decides whether to build) and does NOT
    connect to Redis (the backend connects lazily on first use)."""
    from cache._redisvl_backend import RedisVLSemanticCache

    answer = RedisVLSemanticCache(index_name="rag_cache_answer", settings=settings)
    retrieval = RedisVLSemanticCache(index_name="rag_cache_retrieval", settings=settings)
    return answer, retrieval
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cache_serialization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cache/__init__.py cache/semantic_cache.py tests/test_cache_serialization.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(cache): SemanticCache protocol, payload serialization, build_cache"
```

---

### Task 3: `FakeSemanticCache` + contract tests

**Files:**
- Create: `tests/cache/__init__.py` (empty)
- Create: `tests/cache/fake_cache.py`
- Test: `tests/test_cache_semantics.py`

**Interfaces:**
- Consumes: `cache.semantic_cache.norm_collection`.
- Produces: `class FakeSemanticCache` implementing `SemanticCache`, constructor `FakeSemanticCache(*, threshold: float = 0.9, ttl_seconds: int = 3600, time_fn: Callable[[], float] = time.time)`. In-memory, brute-force cosine, TAG-style scoping, targeted eviction, injectable-clock TTL.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_semantics.py
from tests.cache.fake_cache import FakeSemanticCache


def _clock():
    box = {"t": 1000.0}
    return box, (lambda: box["t"])


def test_hit_within_threshold_miss_outside():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    # identical vector -> hit
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) == {"v": 1}
    # orthogonal vector (cosine 0) -> miss
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0]) is None


def test_never_crosses_tenant():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    assert c.lookup(tenant_id="other", collection_id=None, embedding=[1.0, 0.0]) is None


def test_never_crosses_collection():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id="kb", embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    # unscoped query must not read the collection-scoped entry
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None


def test_targeted_eviction_removes_only_referencing_entries():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1", "d2"])
    c.store(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0],
            payload={"v": 2}, doc_ids=["d3"])
    n = c.invalidate_document(tenant_id="acme", collection_id=None, doc_id="d1")
    assert n == 1
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0]) == {"v": 2}


def test_ttl_expiry_via_injected_clock():
    box, now = _clock()
    c = FakeSemanticCache(threshold=0.9, ttl_seconds=100, time_fn=now)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    box["t"] = 1050.0  # within TTL
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) == {"v": 1}
    box["t"] = 1101.0  # past TTL
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_semantics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.cache'`.

- [ ] **Step 3: Implement the fake**

`tests/cache/__init__.py`: empty.

`tests/cache/fake_cache.py`:

```python
"""In-memory SemanticCache for the offline suite: brute-force cosine, TAG-style
tenant/collection scoping, targeted eviction, injectable-clock TTL. No Redis."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from cache.semantic_cache import norm_collection


@dataclass
class _Entry:
    embedding: list[float]
    payload: dict
    doc_ids: set[str]
    expires_at: float


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class FakeSemanticCache:
    threshold: float = 0.9
    ttl_seconds: int = 3600
    time_fn: Callable[[], float] = time.time
    _store: dict[tuple[str, str], list[_Entry]] = field(default_factory=dict)

    def _key(self, tenant_id: str, collection_id: str | None) -> tuple[str, str]:
        return (tenant_id, norm_collection(collection_id))

    def _live(self, bucket: list[_Entry]) -> list[_Entry]:
        now = self.time_fn()
        return [e for e in bucket if e.expires_at > now]

    def lookup(self, *, tenant_id, collection_id, embedding):
        bucket = self._store.get(self._key(tenant_id, collection_id), [])
        best, best_sim = None, -1.0
        for e in self._live(bucket):
            sim = _cosine(embedding, e.embedding)
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= self.threshold:
            return best.payload
        return None

    def store(self, *, tenant_id, collection_id, embedding, payload, doc_ids):
        bucket = self._store.setdefault(self._key(tenant_id, collection_id), [])
        bucket.append(_Entry(
            embedding=list(embedding), payload=payload, doc_ids=set(doc_ids),
            expires_at=self.time_fn() + self.ttl_seconds,
        ))

    def invalidate_document(self, *, tenant_id, collection_id, doc_id) -> int:
        key = self._key(tenant_id, collection_id)
        bucket = self._store.get(key, [])
        keep = [e for e in bucket if doc_id not in e.doc_ids]
        removed = len(bucket) - len(keep)
        self._store[key] = keep
        return removed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cache_semantics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/cache/__init__.py tests/cache/fake_cache.py tests/test_cache_semantics.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "test(cache): FakeSemanticCache + threshold/isolation/eviction/TTL contract tests"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `core/pipeline.py` — `RAGPipeline.__init__`, `RAGPipeline.answer`, `build()`
- Test: `tests/test_cache_pipeline.py`

**Interfaces:**
- Consumes: `cache.semantic_cache` helpers; `tests.cache.fake_cache.FakeSemanticCache`; `Settings.cache_enabled`.
- Produces: `RAGPipeline(..., embedder=None, answer_cache=None, retrieval_cache=None)`; `build(..., enable_cache: bool | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_pipeline.py
from cache.semantic_cache import answer_to_payload, scored_to_payload
from core.pipeline import RAGPipeline
from core.types import Answer, Chunk, ScoredChunk, Usage
from tests.cache.fake_cache import FakeSemanticCache


class _Embedder:
    model = "e"
    def embed_query(self, text): return [1.0, 0.0]
    def embed_documents(self, ts): return [[1.0, 0.0] for _ in ts]


class _Retriever:
    def __init__(self):
        self.embedder = _Embedder()
        self.calls = 0
    def retrieve(self, query):
        self.calls += 1
        return [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx",
                            tenant_id="public"), score=0.9)]


class _Grounded:
    def __init__(self):
        self.calls = 0
    def generate(self, question, scored):
        self.calls += 1
        return Answer(text=f"ans:{question}", refused=False, usage=Usage())


def _settings():
    from core.config import Settings
    return Settings()


def _pipe(answer_cache, retrieval_cache):
    ret, gen = _Retriever(), _Grounded()
    p = RAGPipeline(ret, gen, _settings(), embedder=ret.embedder,
                    answer_cache=answer_cache, retrieval_cache=retrieval_cache)
    return p, ret, gen


def test_answer_hit_skips_retrieval_and_generation():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    ac.store(tenant_id="public", collection_id=None, embedding=[1.0, 0.0],
             payload=answer_to_payload(Answer(text="cached", refused=False)),
             doc_ids=["d1"])
    p, ret, gen = _pipe(ac, rc)
    ans = p.answer("hello")
    assert ans.text == "cached"
    assert ret.calls == 0 and gen.calls == 0


def test_retrieval_hit_skips_retrieval_but_generates():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    sc = [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx",
          tenant_id="public"), score=0.9)]
    rc.store(tenant_id="public", collection_id=None, embedding=[1.0, 0.0],
             payload=scored_to_payload(sc), doc_ids=["d1"])
    p, ret, gen = _pipe(ac, rc)
    ans = p.answer("hello")
    assert ret.calls == 0 and gen.calls == 1
    assert ans.text == "ans:hello"


def test_full_miss_runs_both_and_populates_tiers():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    p, ret, gen = _pipe(ac, rc)
    p.answer("hello")
    assert ret.calls == 1 and gen.calls == 1
    # both tiers now warm for the same query
    assert rc.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is not None
    assert ac.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is not None


def test_refused_answer_is_not_cached():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    ret = _Retriever()
    class _RefusingGen:
        def generate(self, q, s): return Answer(text="no", refused=True, usage=Usage())
    p = RAGPipeline(ret, _RefusingGen(), _settings(), embedder=ret.embedder,
                    answer_cache=ac, retrieval_cache=rc)
    p.answer("hello")
    assert ac.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is None


def test_no_cache_wired_is_a_total_bypass():
    ret, gen = _Retriever(), _Grounded()
    p = RAGPipeline(ret, gen, _settings(), embedder=ret.embedder,
                    answer_cache=None, retrieval_cache=None)
    ans = p.answer("hello")
    assert ret.calls == 1 and gen.calls == 1 and ans.text == "ans:hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_pipeline.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'embedder'`.

- [ ] **Step 3: Wire the cache into `RAGPipeline`**

In `core/pipeline.py`, add the imports near the top:

```python
from cache.semantic_cache import (
    answer_from_payload, answer_to_payload, doc_ids_of,
    scored_from_payload, scored_to_payload,
)
```

Extend `__init__` signature and body (keep existing params/lines; add the four new lines):

```python
    def __init__(
        self,
        retriever,
        grounded: GroundedGenerator,
        settings: Settings,
        tracer: Tracer | None = None,
        guardrails: GuardrailRunner | None = None,
        embedder=None,
        answer_cache=None,
        retrieval_cache=None,
    ):
        self.retriever = retriever
        self.grounded = grounded
        self.settings = settings
        self.default_acl = ACLContext(tenant_id=DEFAULT_TENANT)
        self.tracer = tracer or Tracer(settings)
        self.guardrails = guardrails
        self.embedder = embedder
        self.answer_cache = answer_cache
        self.retrieval_cache = retrieval_cache
```

In `answer()`, immediately AFTER the root span is opened (`with self.tracer.span("rag.query", ...) as root:`) and after the guardrail-input trace block, compute the key vector and consult the answer tier BEFORE building `Query`. Insert:

```python
            # --- Semantic cache: answer tier -----------------------------------
            cache_on = self.answer_cache is not None or self.retrieval_cache is not None
            key_vec = None
            if cache_on and self.embedder is not None:
                key_vec = self.embedder.embed_query(question)
            if key_vec is not None and self.answer_cache is not None:
                hit = self.answer_cache.lookup(
                    tenant_id=acl.tenant_id, collection_id=collection_id, embedding=key_vec)
                if hit is not None:
                    root.update(output={"cache": "answer_hit"})
                    cached = answer_from_payload(hit)
                    cached.metadata.setdefault("stage_latencies_ms", {})
                    cached.metadata["cache"] = "answer_hit"
                    return cached
```

Replace the retrieval span body so it consults the retrieval tier before retrieving. Change the block that currently reads `scored, ms = timed(self.retriever.retrieve)(q)` to:

```python
            with self.tracer.span("retrieval", top_k=q.top_k) as s_ret:
                scored = None
                if key_vec is not None and self.retrieval_cache is not None:
                    rhit = self.retrieval_cache.lookup(
                        tenant_id=acl.tenant_id, collection_id=collection_id, embedding=key_vec)
                    if rhit is not None:
                        scored = scored_from_payload(rhit)
                        latencies["retrieval_ms"] = 0.0
                        s_ret.update(output={"cache": "retrieval_hit", "n_hits": len(scored)})
                if scored is None:
                    scored, ms = timed(self.retriever.retrieve)(q)
                    latencies["retrieval_ms"] = ms
                    s_ret.update(output={"n_hits": len(scored)})
                    if key_vec is not None and self.retrieval_cache is not None:
                        self.retrieval_cache.store(
                            tenant_id=acl.tenant_id, collection_id=collection_id,
                            embedding=key_vec, payload=scored_to_payload(scored),
                            doc_ids=doc_ids_of(scored))
                suspected = sorted({
                    lbl for sc in scored for lbl in scan_for_injection(sc.chunk.text)
                })
                if suspected:
                    s_ret.update(output={"indirect_injection_suspected": suspected})
                    logger.warning("indirect_injection_suspected: %s", suspected)
```

Finally, after the output-guardrail block and AFTER `blocked_by`/refusal is resolved (just before the `cost = cost_usd(...)` line, or immediately after computing `ans.refused`), store the answer only when clean. Insert right before `return ans`, after all metadata is attached but guarded on state:

```python
        # --- Semantic cache: store the answer only when fully clean -----------
        if (key_vec is not None and self.answer_cache is not None
                and not ans.refused
                and ans.metadata.get("blocked_by") != "output_guardrail"):
            self.answer_cache.store(
                tenant_id=acl.tenant_id, collection_id=collection_id,
                embedding=key_vec, payload=answer_to_payload(ans),
                doc_ids=ans.metadata.get("retrieved_doc_ids", []))
```

> Placement note: `key_vec` is assigned inside the `with ... as root:` block. Because Python `with` does not create a new scope, `key_vec` is still in scope at the `return ans` at the end of `answer()`. Verify by reading the method after editing.

Update `build()` — add the `enable_cache` parameter and wire the tiers + embedder:

```python
def build(
    version: str = "full",
    corpus: str | None = None,
    settings: Settings | None = None,
    enable_guardrails: bool | None = None,
    enable_cache: bool | None = None,
) -> RAGPipeline:
```

At the end of `build()`, replace the final `return RAGPipeline(...)` with:

```python
    use_guards = s.guardrails_enabled if enable_guardrails is None else enable_guardrails
    guardrails = default_runner(generator=generator) if use_guards else None

    use_cache = s.cache_enabled if enable_cache is None else enable_cache
    answer_cache = retrieval_cache = None
    if use_cache:
        from cache.semantic_cache import build_cache
        answer_cache, retrieval_cache = build_cache(s)

    return RAGPipeline(retriever, grounded, s, guardrails=guardrails,
                       embedder=embedder, answer_cache=answer_cache,
                       retrieval_cache=retrieval_cache)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cache_pipeline.py -v`
Expected: PASS (5 tests).

Then run the existing pipeline suite to confirm no regression:
Run: `.venv/bin/python -m pytest tests/test_pipeline_integration.py -v`
Expected: PASS (cache is off by default — `embedder`/caches default `None`).

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/test_cache_pipeline.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(cache): cache-aware RAGPipeline.answer with answer+retrieval tiers"
```

---

### Task 5: Worker invalidation

**Files:**
- Modify: `ingest/worker.py` — `IngestDeps`, `_build_deps`, `run_ingest`, `run_delete`
- Test: `tests/test_cache_worker.py`

**Interfaces:**
- Consumes: `cache.semantic_cache.build_cache`; `Settings.cache_enabled`.
- Produces: `IngestDeps.caches: tuple | None`; both worker bodies call `invalidate_document` on each tier after their commit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_worker.py
from dataclasses import dataclass

from ingest.worker import IngestDeps, run_delete, run_ingest


class _RecordingCache:
    def __init__(self): self.calls = []
    def lookup(self, **kw): return None
    def store(self, **kw): pass
    def invalidate_document(self, *, tenant_id, collection_id, doc_id):
        self.calls.append((tenant_id, collection_id, doc_id)); return 1


@dataclass
class _Rec:
    tenant_id: str = "acme"
    collection_id: str = "kb"
    content_type: str = "text/plain"
    filename: str = "f.txt"
    blob_key: str = "acme/f"


class _Registry:
    def __init__(self, rec): self._rec = rec
    def get_privileged(self, _id): return self._rec
    def set_status(self, *a, **k): pass
    def delete(self, *a, **k): pass


class _Ingestor:
    def ingest_document(self, *a, **k): return 3
    def delete_document(self, *a, **k): pass


class _Blobs:
    def get(self, key): return b"hello"
    def delete(self, key): pass


class _Parser:
    def parse(self, *a, **k):
        from core.types import Document
        return [Document(doc_id="d1", text="hello", tenant_id="acme")]


class _Parsers:
    def resolve(self, _ct): return _Parser()


def _deps(caches, settings):
    return IngestDeps(registry=_Registry(_Rec()), blobs=_Blobs(), parsers=_Parsers(),
                      ingestor=_Ingestor(), settings=settings, caches=caches)


def _settings():
    from core.config import Settings
    return Settings()


def test_run_delete_invalidates_both_tiers():
    a, r = _RecordingCache(), _RecordingCache()
    run_delete(_deps((a, r), _settings()), "d1")
    assert a.calls == [("acme", "kb", "d1")]
    assert r.calls == [("acme", "kb", "d1")]


def test_run_ingest_invalidates_both_tiers():
    a, r = _RecordingCache(), _RecordingCache()
    run_ingest(_deps((a, r), _settings()), "d1")
    assert a.calls == [("acme", "kb", "d1")]
    assert r.calls == [("acme", "kb", "d1")]


def test_no_caches_is_noop():
    # caches=None must not raise
    run_delete(_deps(None, _settings()), "d1")
```

> Note: read `core/types.py` for the real `Document` constructor before writing `_Parser.parse`; adjust kwargs to the minimal valid set. The assertions on `caches` are the point.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_worker.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'caches'`.

- [ ] **Step 3: Add `caches` to deps and invalidate after commit**

In `ingest/worker.py`, extend the dataclass:

```python
@dataclass
class IngestDeps:
    registry: object   # DocumentRegistry
    blobs: object      # BlobStore
    parsers: object    # ParserRegistry
    ingestor: object   # IncrementalIngestor
    settings: Settings
    caches: object = None   # tuple[SemanticCache, SemanticCache] | None
```

Add a private helper below `_pii_process`:

```python
def _invalidate_caches(deps: IngestDeps, tenant_id: str, collection_id, doc_id: str) -> None:
    """Evict every cached answer/retrieval that cites this document, both tiers.
    No-op when the cache is disabled. Never raises into the worker body."""
    if not deps.caches:
        return
    for cache in deps.caches:
        try:
            cache.invalidate_document(
                tenant_id=tenant_id, collection_id=collection_id or None, doc_id=doc_id)
        except Exception:  # cache eviction must never fail the ingest/delete
            logger.exception("cache invalidation failed for %s", doc_id)
```

In `run_ingest`, after the `set_status(... READY ...)` line inside the `try`:

```python
        n = deps.ingestor.ingest_document(tenant_id, document_id, chunks, acl)
        deps.registry.set_status(document_id, tenant_id, DocumentStatus.READY, chunk_count=n)
        _invalidate_caches(deps, tenant_id, rec.collection_id, document_id)
```

In `run_delete`, after `deps.registry.delete(...)`:

```python
        deps.ingestor.delete_document(tenant_id, document_id, acl)
        deps.blobs.delete(rec.blob_key)
        deps.registry.delete(document_id, tenant_id)
        _invalidate_caches(deps, tenant_id, rec.collection_id, document_id)
```

Wire it into `_build_deps`:

```python
def _build_deps(settings: Settings) -> IngestDeps:
    from core.registry import (build_blob_store, build_document_registry,
                               build_incremental_ingestor, build_parser_registry)
    caches = None
    if settings.cache_enabled:
        from cache.semantic_cache import build_cache
        caches = build_cache(settings)
    return IngestDeps(
        registry=build_document_registry(settings),
        blobs=build_blob_store(settings),
        parsers=build_parser_registry(settings),
        ingestor=build_incremental_ingestor(settings),
        settings=settings,
        caches=caches,
    )
```

> `rec.collection_id` may be `""`; `_invalidate_caches` maps falsy to `None` so the backend's `norm_collection` produces the same sentinel the query path used.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cache_worker.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ingest/worker.py tests/test_cache_worker.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(cache): workers evict cached entries for changed/deleted docs"
```

---

### Task 6: redis-vl backend + import-isolation guard + live smoke

**Files:**
- Create: `cache/_redisvl_backend.py`
- Test: `tests/test_cache_backend.py` (offline: import isolation + construction)
- Test: `tests/test_cache_live_smoke.py` (opt-in live round-trip)

**Interfaces:**
- Consumes: `cache.semantic_cache.norm_collection`; `Settings.redis_url`, `redis_password`, `embed_dimension`, `cache_similarity_threshold`, `cache_ttl_seconds`.
- Produces: `class RedisVLSemanticCache` implementing `SemanticCache`; `__init__(self, *, index_name: str, settings: Settings)` stores config only (no import, no connection).

- [ ] **Step 1: Write the failing offline test**

```python
# tests/test_cache_backend.py
import sys

import cache.semantic_cache as sc


def test_importing_cache_does_not_import_redisvl():
    # Neither the seam nor the backend module may pull redis-vl at import time.
    import cache._redisvl_backend  # noqa: F401
    assert "redisvl" not in sys.modules
    assert "redis_vl" not in sys.modules


def test_build_cache_constructs_two_named_tiers_without_connecting():
    from core.config import Settings
    answer, retrieval = sc.build_cache(Settings())
    assert answer.index_name == "rag_cache_answer"
    assert retrieval.index_name == "rag_cache_retrieval"
    # Construction must not require Redis or redis-vl to be installed/reachable.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cache_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cache._redisvl_backend'`.

- [ ] **Step 3: Implement the backend (all redis-vl imports lazy, inside methods)**

`cache/_redisvl_backend.py`:

```python
"""redis-vl-backed SemanticCache. The ONLY module importing redis-vl, and only
lazily inside method bodies — constructing an instance requires neither Redis nor
the redis-vl package, so build_cache stays importable in the offline suite.

Schema per tier (one RediSearch index): a COSINE vector field plus tenant_id,
collection_id and doc_ids TAG fields and a payload text field. The doc_ids TAG is
the reverse index that makes per-document eviction a filtered delete. Per-key TTL
backstops the new-document blind spot.
"""

from __future__ import annotations

import json
from typing import Sequence

from cache.semantic_cache import norm_collection


class RedisVLSemanticCache:
    def __init__(self, *, index_name: str, settings) -> None:
        self.index_name = index_name
        self.settings = settings
        self._index = None  # lazily built SearchIndex

    def _get_index(self):
        if self._index is not None:
            return self._index
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_dict({
            "index": {"name": self.index_name, "prefix": f"{self.index_name}:",
                      "storage_type": "hash"},
            "fields": [
                {"name": "tenant_id", "type": "tag"},
                {"name": "collection_id", "type": "tag"},
                {"name": "doc_ids", "type": "tag", "attrs": {"separator": "|"}},
                {"name": "payload", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {
                    "dims": self.settings.embed_dimension, "distance_metric": "cosine",
                    "algorithm": "flat", "datatype": "float32"}},
            ],
        })
        index = SearchIndex(schema, redis_url=self.settings.redis_url)
        index.create(overwrite=False)
        self._index = index
        return index

    def _distance_threshold(self) -> float:
        # redis-vl ranges cosine DISTANCE in [0, 2]; distance = 1 - similarity.
        return 1.0 - float(self.settings.cache_similarity_threshold)

    def lookup(self, *, tenant_id, collection_id, embedding) -> dict | None:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        index = self._get_index()
        flt = (Tag("tenant_id") == tenant_id) & \
              (Tag("collection_id") == norm_collection(collection_id))
        q = VectorQuery(vector=list(embedding), vector_field_name="vector",
                        return_fields=["payload", "vector_distance"], num_results=1,
                        filter_expression=flt)
        results = index.query(q)
        if not results:
            return None
        top = results[0]
        if float(top["vector_distance"]) > self._distance_threshold():
            return None
        return json.loads(top["payload"])

    def store(self, *, tenant_id, collection_id, embedding, payload, doc_ids) -> None:
        import numpy as np

        index = self._get_index()
        vec = np.array(list(embedding), dtype=np.float32).tobytes()
        data = {
            "tenant_id": tenant_id,
            "collection_id": norm_collection(collection_id),
            "doc_ids": "|".join(doc_ids) if doc_ids else "",
            "payload": json.dumps(payload),
            "vector": vec,
        }
        index.load([data], ttl=int(self.settings.cache_ttl_seconds))

    def invalidate_document(self, *, tenant_id, collection_id, doc_id) -> int:
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        index = self._get_index()
        flt = (Tag("tenant_id") == tenant_id) & \
              (Tag("collection_id") == norm_collection(collection_id)) & \
              (Tag("doc_ids") == doc_id)
        matches = index.query(FilterQuery(filter_expression=flt, return_fields=["id"]))
        keys = [m["id"] for m in matches]
        if keys:
            index.drop_keys(keys)
        return len(keys)
```

> The exact redis-vl call surface (`index.load(..., ttl=)`, `drop_keys`, `VectorQuery`/`FilterQuery` field names) is from redis-vl's current API — verify against the installed version via context7/docs during implementation and adjust names if the version differs. The offline test only asserts import isolation + construction; the live smoke (below) is what proves the calls.

- [ ] **Step 4: Run the offline test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cache_backend.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the opt-in live smoke test**

```python
# tests/test_cache_live_smoke.py
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CACHE_LIVE_SMOKE") != "1",
    reason="set CACHE_LIVE_SMOKE=1 with a running Redis 8 to exercise redis-vl",
)


def test_redisvl_store_lookup_invalidate_round_trip():
    from cache._redisvl_backend import RedisVLSemanticCache
    from core.config import Settings

    s = Settings(cache_enabled=True, embed_dimension=4, cache_similarity_threshold=0.9)
    c = RedisVLSemanticCache(index_name="rag_cache_smoke", settings=s)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0, 0.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    assert c.lookup(tenant_id="acme", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) == {"v": 1}
    assert c.lookup(tenant_id="other", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) is None
    assert c.invalidate_document(tenant_id="acme", collection_id=None, doc_id="d1") == 1
    assert c.lookup(tenant_id="acme", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) is None
```

- [ ] **Step 6: Verify the live smoke skips cleanly offline**

Run: `.venv/bin/python -m pytest tests/test_cache_live_smoke.py -v`
Expected: 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add cache/_redisvl_backend.py tests/test_cache_backend.py tests/test_cache_live_smoke.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(cache): redis-vl backend (lazy) + import-isolation guard + live smoke"
```

---

### Task 7: Infra, dependency, and docs

**Files:**
- Modify: `infra/docker-compose.yml` (`redis:7` → `redis:8`)
- Modify: `pyproject.toml` (add `cache` extra with `redis-vl`); regenerate `uv.lock`
- Modify: `.env.example`
- Modify: `docs/architecture.md`, `docs/PROJECT_STATUS.md`

**Interfaces:**
- Consumes: nothing new. Produces: runnable infra + documented feature. No new Python symbols.

- [ ] **Step 1: Bump Redis image**

In `infra/docker-compose.yml`, find the app Redis service (the one on `redis://localhost:6379` used by arq — NOT any Langfuse-stack redis) and change its image tag from `redis:7*` to `redis:8-alpine`. Leave ports/volumes unchanged.

Verify: `docker compose -f infra/docker-compose.yml config >/dev/null && echo OK`
Expected: `OK` (compose still parses).

- [ ] **Step 2: Add the redis-vl dependency extra**

In `pyproject.toml`, add a `cache` optional-dependency group:

```toml
[project.optional-dependencies]
cache = ["redis-vl>=0.3"]
```

(If `[project.optional-dependencies]` already exists, add the `cache` line into it and append `"redis-vl>=0.3"` to the `all` extra if one is maintained.)

Regenerate the lock: `uv lock`
Verify it still resolves: `uv lock --check` → expected: no error.

- [ ] **Step 3: Document env vars**

Append to `.env.example` under an ingest/cache section:

```bash
# --- Semantic cache (Redis 8 / redis-vl; opt-in) ---
CACHE_ENABLED=false
CACHE_SIMILARITY_THRESHOLD=0.9
CACHE_TTL_SECONDS=3600
```

- [ ] **Step 4: Document the feature**

Add a "Semantic cache" subsection to `docs/architecture.md` (near the query-path / retrieval description) summarizing: two tiers (answer + retrieval), semantic match via redis-vl on Redis 8, tenant/collection isolation, per-document targeted eviction + TTL backstop, eval bypass, opt-in default. Update `docs/PROJECT_STATUS.md` to list Decomposition E as implemented (plumbing + fake + live smoke; live baseline enabling deferred).

- [ ] **Step 5: Run the full offline suite**

Run: `.venv/bin/python -m pytest tests/ -p no:warnings -q`
Expected: all pass (new cache tests green; live smoke skipped); no import errors from the new package. Confirm exit code 0.

- [ ] **Step 6: Commit**

```bash
git add infra/docker-compose.yml pyproject.toml uv.lock .env.example docs/architecture.md docs/PROJECT_STATUS.md
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "chore(cache): Redis 8 image, redis-vl extra, env + architecture docs"
```

---

## Final verification (after all tasks)

- [ ] Full suite green: `.venv/bin/python -m pytest tests/ -p no:warnings -q` (exit 0).
- [ ] Lint clean on new files: `.venv/bin/python -m ruff check cache/ tests/test_cache_*.py tests/cache/`.
- [ ] Import isolation holds: `.venv/bin/python -c "import cache.semantic_cache, cache._redisvl_backend, sys; assert 'redisvl' not in sys.modules"`.
- [ ] Whole-branch review against the 7 spec invariants (§12 of the design).

## Self-Review (author checklist — completed)

**Spec coverage:** Answer tier + retrieval tier (Task 4); skip chunk cache (design, no task needed); semantic match (Tasks 3/6); redis-vl on Redis 8 (Tasks 6/7); targeted eviction via TAG reverse index (Tasks 3/5/6); TTL backstop (Tasks 1/3/6); tenant+collection isolation (Tasks 3/6, tested); eval bypass (Task 4 `enable_cache`); refusals not cached (Task 4, tested); config knobs (Task 1); infra + deps + docs (Task 7); import isolation (Task 6, tested); opt-in default (Task 1). All §11 files covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Two "verify against real API/types" notes (redis-vl call surface, `Citation`/`Document` constructors) are deliberate implementation checks, not placeholders — the surrounding code is complete and the assertions are exact.

**Type consistency:** `SemanticCache` method signatures identical across Protocol (Task 2), fake (Task 3), backend (Task 6), and consumers (Tasks 4/5). `build_cache` returns `(answer, retrieval)` everywhere. `norm_collection`/`COLLECTION_NONE` used identically in fake and backend. `answer_to_payload`/`scored_to_payload` and their inverses match across Tasks 2/4.
