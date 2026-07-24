# SP12 · Native Query Rewriting & Synonym Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a two-tiered, tenant-safe query rewriter — deterministic per-tenant Redis synonym substitution with a structured LLM query-expansion fallback — wired into the `RAGPipeline` query path so retrieval benefits from expanded queries while generation stays faithful to the user's original question.

**Architecture:** Add a `QueryRewriter` Protocol; build a `HybridQueryRewriter` (Redis synonym hash + cheap `Generator`); inject it into `RAGPipeline` via `build()` (dependency-injection, not constructed in `__init__`). The rewrite runs **after** input-guardrail redaction and **before** the `key_vec` embed, so both semantic-cache tiers key on the rewritten query (spec G3). Generation and output-guard context use the **original** redacted question.

**Tech Stack:** Python 3.11–3.13, Pydantic, redis (lazily imported; no top-level import, no `fakeredis` dependency — tests inject a fake client).

> **NOTE — this plan was rewritten 2026-07-24** to reconcile the original 2026-07-12 draft with the current codebase (post product-pivot and post Decomposition E semantic cache). Corrections vs the original: (1) reuse `settings.redis_url` instead of a new `rewriter_redis_url`; (2) rewriter is an injected `RAGPipeline` constructor param built in `build()`, not built inside `__init__`; (3) rewrite lands before the `key_vec` embed that both cache tiers use; (4) retrieval uses the rewritten query, generation uses the original; (5) redis is injected/lazy to keep the offline suite infra-free; (6) eval wiring added for G4.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- Tenant synonym isolation is hostile-by-design: synonym keys partition strictly on `rewriter:synonyms:{tenant_id}` namespaces; a tenant's dictionary can never affect another tenant's rewrite.
- Rewriting failures (Redis unreachable, LLM error) degrade immediately and silently to the best-effort query (synonym result if any, else raw) — never raise into the query path (fail-soft).
- Offline-safe: importing `providers/rewriter/hybrid_rewriter.py` must not require Redis or a live `redis` connection. No top-level `import redis`.
- Rollout posture **C** (approved): `rewriter_enabled=True`, `rewriter_llm_enabled=True`, `rewriter_llm_threshold=5`.
- Test runner: `.venv/bin/python -m pytest <path> -v` (exit 0 = pass).

---

### Task 1: Configuration knobs

**Files:**
- Modify: `core/config.py`
- Test: `tests/test_sp12_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings.rewriter_enabled: bool`, `Settings.rewriter_llm_enabled: bool`, `Settings.rewriter_llm_threshold: int`. (Redis URL is the existing `settings.redis_url` — do NOT add a new url knob.)

- [ ] **Step 1: Write the failing test** — `tests/test_sp12_config.py`
```python
from core.config import Settings


def test_sp12_rewriter_defaults():
    s = Settings()
    assert s.rewriter_enabled is True
    assert s.rewriter_llm_enabled is True
    assert s.rewriter_llm_threshold == 5
    # Reuses the existing shared Redis URL — no dedicated rewriter url knob.
    assert not hasattr(s, "rewriter_redis_url")
    assert s.redis_url.startswith("redis://")
```

- [ ] **Step 2: Run to verify it fails**
Run: `.venv/bin/python -m pytest tests/test_sp12_config.py -v`
Expected: FAIL (AttributeError: rewriter_enabled).

- [ ] **Step 3: Implement** — add under `Settings` in `core/config.py`, near the semantic-cache block:
```python
    # --- Query rewriting & synonym expansion (SP12) ---
    # Runs after input-guard redaction, before the cache-key embed. Synonym tier
    # reads rewriter:synonyms:{tenant_id} from the shared Redis (redis_url).
    rewriter_enabled: bool = True
    rewriter_llm_enabled: bool = True
    rewriter_llm_threshold: int = 5
```

- [ ] **Step 4: Run to verify it passes**
Run: `.venv/bin/python -m pytest tests/test_sp12_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add core/config.py tests/test_sp12_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): add query-rewriter config knobs (SP12)"
```

---

### Task 2: `QueryRewriter` Protocol

**Files:**
- Modify: `core/interfaces.py`
- Test: `tests/test_sp12_interface.py`

**Interfaces:**
- Consumes: `ACLContext` (already imported in `core/interfaces.py`)
- Produces: `QueryRewriter` Protocol with `rewrite(self, query: str, acl: ACLContext) -> str`

- [ ] **Step 1: Write the failing test** — `tests/test_sp12_interface.py`
```python
from core.interfaces import QueryRewriter


def test_query_rewriter_protocol_shape():
    assert hasattr(QueryRewriter, "rewrite")

    class _Ok:
        def rewrite(self, query, acl):
            return query

    assert isinstance(_Ok(), QueryRewriter)
```

- [ ] **Step 2: Run to verify it fails**
Run: `.venv/bin/python -m pytest tests/test_sp12_interface.py -v`
Expected: FAIL (ImportError: cannot import name 'QueryRewriter').

- [ ] **Step 3: Implement** — add to `core/interfaces.py` (match the file's existing `@runtime_checkable` convention):
```python
@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrites/expands a query for retrieval. MUST be fail-soft: any internal
    error returns a best-effort query rather than raising into the query path."""

    def rewrite(self, query: str, acl: ACLContext) -> str: ...
```

- [ ] **Step 4: Run to verify it passes**
Run: `.venv/bin/python -m pytest tests/test_sp12_interface.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py tests/test_sp12_interface.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): define QueryRewriter protocol (SP12)"
```

---

### Task 3: `HybridQueryRewriter` + registry factory

**Files:**
- Create: `providers/rewriter/__init__.py`, `providers/rewriter/hybrid_rewriter.py`
- Modify: `core/registry.py`
- Test: `tests/test_sp12_rewriter.py`

**Interfaces:**
- Consumes: `Generator` (via `build_generator("context")`), `ACLContext`, `ChatMessage`, `settings.redis_url`
- Produces: `HybridQueryRewriter(generator, redis_url, *, llm_enabled=True, llm_threshold=5, redis_client=None)`; `core.registry.build_query_rewriter(settings) -> QueryRewriter`

Behavior contract:
- Synonym tier: read hash `rewriter:synonyms:{acl.tenant_id}`; for each `shortcut -> full`, case-insensitive word-boundary regex replace. Decode bytes-or-str keys/values. Sort by descending shortcut length so longer phrases win before their substrings.
- LLM tier: trigger only when `llm_enabled and len(query.split()) >= llm_threshold and not synonym_replaced`. One `generator.complete([system, user], max_tokens=128, temperature=0.0)`; return `resp.text.strip()` (fall back to the synonym/raw result if the LLM returns empty).
- Fail-soft: wrap Redis access and the LLM call each in `try/except Exception` + `logger.warning/error`; on failure keep the best result so far.
- Offline-safe: `import redis` happens lazily inside a memoized `_client()` method, only when `redis_client` was not injected. Constructing the object touches no network.

- [ ] **Step 1: Write the failing tests** — `tests/test_sp12_rewriter.py`
```python
import pytest

from core.types import ACLContext
from providers.rewriter.hybrid_rewriter import HybridQueryRewriter
from tests._fakes import RecordingGenerator


class FakeRedis:
    """Minimal hgetall-only stand-in; maps key -> {field: value}."""

    def __init__(self, data):
        self._data = data

    def hgetall(self, key):
        return dict(self._data.get(key, {}))


def _acl(t="t1"):
    return ACLContext(tenant_id=t)


def test_synonym_substitution_word_boundary():
    r = FakeRedis({"rewriter:synonyms:t1": {"NYPD": "New York Police Department"}})
    gen = RecordingGenerator(text="UNUSED")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=False, redis_client=r)
    assert rw.rewrite("who leads NYPD?", _acl("t1")) == "who leads New York Police Department?"
    # Substring guard: "NYPDX" must not match.
    assert rw.rewrite("NYPDX status", _acl("t1")) == "NYPDX status"


def test_tenant_synonym_isolation():
    r = FakeRedis({"rewriter:synonyms:t2": {"Jupiter": "Jupiter-Next"}})
    gen = RecordingGenerator(text="UNUSED")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=False, redis_client=r)
    # Tenant t1 has no dictionary — t2's mapping must not leak.
    assert rw.rewrite("Jupiter roadmap", _acl("t1")) == "Jupiter roadmap"
    assert rw.rewrite("Jupiter roadmap", _acl("t2")) == "Jupiter-Next roadmap"


def test_llm_expansion_triggers_only_over_threshold_without_synonym():
    r = FakeRedis({})
    gen = RecordingGenerator(text="expanded descriptive query")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=True, llm_threshold=5, redis_client=r)
    # 3 words < threshold -> no LLM, raw returned.
    assert rw.rewrite("short simple query", _acl()) == "short simple query"
    assert gen.calls == 0
    # 5 words >= threshold, no synonym match -> LLM expansion.
    assert rw.rewrite("please find the sales reports", _acl()) == "expanded descriptive query"
    assert gen.calls == 1


def test_synonym_match_suppresses_llm():
    r = FakeRedis({"rewriter:synonyms:t1": {"reports": "quarterly financial reports"}})
    gen = RecordingGenerator(text="SHOULD NOT RUN")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=True, llm_threshold=3, redis_client=r)
    out = rw.rewrite("please find the sales reports", _acl("t1"))
    assert "quarterly financial reports" in out
    assert gen.calls == 0


def test_fail_soft_on_redis_error():
    class BoomRedis:
        def hgetall(self, key):
            raise RuntimeError("redis down")

    gen = RecordingGenerator(text="UNUSED")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=False, redis_client=BoomRedis())
    assert rw.rewrite("hello world", _acl()) == "hello world"  # no raise, raw returned


def test_fail_soft_on_llm_error():
    class BoomGen:
        def complete(self, *a, **k):
            raise RuntimeError("llm down")

    r = FakeRedis({})
    rw = HybridQueryRewriter(BoomGen(), "redis://x", llm_enabled=True, llm_threshold=1, redis_client=r)
    assert rw.rewrite("expand me please now", _acl()) == "expand me please now"
```

> Check `tests/_fakes.RecordingGenerator`'s actual constructor/attribute names first. If it doesn't expose a settable canned `text` and a `calls` counter, add a small local fake in this test file instead of forcing the shared fake to change.

- [ ] **Step 2: Run to verify it fails**
Run: `.venv/bin/python -m pytest tests/test_sp12_rewriter.py -v`
Expected: FAIL (ModuleNotFoundError: providers.rewriter).

- [ ] **Step 3: Implement** — `providers/rewriter/__init__.py` (empty) and `providers/rewriter/hybrid_rewriter.py`:
```python
from __future__ import annotations

import logging
import re
from typing import Any

from core.interfaces import Generator
from core.types import ACLContext, ChatMessage

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert search assistant. Rewrite the user's query to maximize "
    "retrieval matching. Return a single descriptive search statement. Keep "
    "specialized terms and expand acronyms. Do not answer the query."
)


def _decode(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


class HybridQueryRewriter:
    """Deterministic per-tenant synonym substitution with an LLM expansion
    fallback. Fail-soft: never raises into the query path."""

    def __init__(
        self,
        generator: Generator,
        redis_url: str,
        *,
        llm_enabled: bool = True,
        llm_threshold: int = 5,
        redis_client: Any | None = None,
    ) -> None:
        self._gen = generator
        self._redis_url = redis_url
        self._llm_enabled = llm_enabled
        self._llm_threshold = llm_threshold
        self._client = redis_client  # injected in tests; lazily built otherwise

    def _get_client(self) -> Any | None:
        if self._client is None:
            try:
                import redis  # lazy: keeps import offline-safe

                self._client = redis.from_url(self._redis_url)
            except Exception as exc:  # pragma: no cover - infra path
                logger.warning("rewriter: redis client init failed: %s", exc)
                return None
        return self._client

    def _synonyms(self, tenant_id: str) -> dict[str, str]:
        client = self._get_client()
        if client is None:
            return {}
        try:
            raw = client.hgetall(f"rewriter:synonyms:{tenant_id}")
        except Exception as exc:
            logger.warning("rewriter: synonym lookup failed: %s", exc)
            return {}
        return {_decode(k): _decode(v) for k, v in (raw or {}).items()}

    def rewrite(self, query: str, acl: ACLContext) -> str:
        rewritten = query
        replaced = False
        syns = self._synonyms(acl.tenant_id)
        # Longer shortcuts first so multi-word phrases win over their substrings.
        for shortcut, full in sorted(syns.items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(rf"\b{re.escape(shortcut)}\b", re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(full, rewritten)
                replaced = True

        if self._llm_enabled and not replaced and len(query.split()) >= self._llm_threshold:
            try:
                resp = self._gen.complete(
                    [
                        ChatMessage(role="system", content=_SYSTEM),
                        ChatMessage(role="user", content=query),
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                expanded = (resp.text or "").strip()
                if expanded:
                    rewritten = expanded
            except Exception as exc:
                logger.error("rewriter: LLM expansion failed, using best-effort: %s", exc)

        return rewritten
```
Add to `core/registry.py` (import `QueryRewriter` in the interfaces import block already present):
```python
def build_query_rewriter(settings: Settings | None = None) -> QueryRewriter:
    s = settings or get_settings()
    from providers.rewriter.hybrid_rewriter import HybridQueryRewriter

    return HybridQueryRewriter(
        build_generator("context", s),
        s.redis_url,
        llm_enabled=s.rewriter_llm_enabled,
        llm_threshold=s.rewriter_llm_threshold,
    )
```

- [ ] **Step 4: Run to verify it passes**
Run: `.venv/bin/python -m pytest tests/test_sp12_rewriter.py -v`
Expected: PASS (all six).

- [ ] **Step 5: Commit**
```bash
git add providers/rewriter/ core/registry.py tests/test_sp12_rewriter.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): HybridQueryRewriter synonym+LLM expander with registry factory (SP12)"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `core/pipeline.py`
- Test: `tests/test_sp12_pipeline.py`

**Interfaces:**
- Consumes: `QueryRewriter`, `build_query_rewriter`
- Produces: `RAGPipeline(..., rewriter=None)` param; `build(..., enable_rewriter: bool | None = None)`; rewrite applied to the **retrieval** query only.

Integration contract (in `RAGPipeline.answer()`):
- Keep `question` = the redacted original throughout generation and output-guard context.
- After input redaction / inside the root span, compute `retrieval_question = self.rewriter.rewrite(question, acl)` when `self.rewriter is not None` (wrapped in a `self.tracer.span("rewrite")`), else `retrieval_question = question`.
- `key_vec` embeds `retrieval_question`; `Query(text=retrieval_question, ...)`.
- `grounded.generate(question, scored)` and the output-guard `"question"` context keep the **original** `question`.

- [ ] **Step 1: Write the failing test** — `tests/test_sp12_pipeline.py`
```python
from core.config import Settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Answer, ScoredChunk, Chunk, Usage


class SpyRewriter:
    def __init__(self):
        self.seen = []

    def rewrite(self, query, acl):
        self.seen.append(query)
        return query + " EXPANDED"


class SpyRetriever:
    def __init__(self):
        self.query_text = None

    def retrieve(self, q):
        self.query_text = q.text
        return [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx"), score=1.0)]


class SpyGrounded:
    def __init__(self):
        self.gen_question = None

    def generate(self, question, scored):
        self.gen_question = question
        return Answer(text="ok", citations=[], contexts=list(scored), usage=Usage())


def test_rewriter_feeds_retrieval_but_not_generation():
    rw, ret, gen = SpyRewriter(), SpyRetriever(), SpyGrounded()
    p = RAGPipeline(ret, gen, Settings(), guardrails=None, embedder=None, rewriter=rw)
    p.answer("original question here", ACLContext(tenant_id="t1"))
    assert rw.seen == ["original question here"]
    assert ret.query_text == "original question here EXPANDED"   # retrieval uses rewrite
    assert gen.gen_question == "original question here"          # generation uses original
```

> Verify the real `ScoredChunk`/`Chunk`/`Answer` constructor field names against `core/types.py` before finalizing — adjust the fakes to match. If constructing `Answer` directly is awkward, reuse an existing pipeline test's helpers/fakes from `tests/`.

- [ ] **Step 2: Run to verify it fails**
Run: `.venv/bin/python -m pytest tests/test_sp12_pipeline.py -v`
Expected: FAIL (TypeError: unexpected keyword 'rewriter').

- [ ] **Step 3: Implement** in `core/pipeline.py`:
  1. Add `rewriter=None` to `RAGPipeline.__init__` params and `self.rewriter = rewriter`.
  2. Inside `answer()`, after the `guardrail.input` span block and before the semantic-cache section, insert:
```python
            retrieval_question = question
            if self.rewriter is not None:
                with self.tracer.span("rewrite") as s_rw:
                    retrieval_question = self.rewriter.rewrite(question, acl)
                    s_rw.update(output={"rewritten": retrieval_question != question})
```
  3. Change the cache-key embed to use `retrieval_question`: `key_vec = self.embedder.embed_query(retrieval_question)`.
  4. Change `Query(text=question, ...)` to `Query(text=retrieval_question, ...)`.
  5. Leave `grounded.generate(question, scored)` and the output-guard `"question": question` context unchanged (original question).
  6. In `build()`: add `enable_rewriter: bool | None = None`; compute `use_rw = s.rewriter_enabled if enable_rewriter is None else enable_rewriter`; `rewriter = build_query_rewriter(s) if use_rw else None` (import `build_query_rewriter` from `core.registry`); pass `rewriter=rewriter` to `RAGPipeline(...)`.

- [ ] **Step 4: Run to verify it passes**
Run: `.venv/bin/python -m pytest tests/test_sp12_pipeline.py -v`
Then the regression guard: `.venv/bin/python -m pytest tests/test_app.py tests/test_pipeline*.py tests/cache -v`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**
```bash
git add core/pipeline.py tests/test_sp12_pipeline.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): wire query rewriter into pipeline (retrieval-only, faithful generation) (SP12)"
```

---

### Task 5: Eval wiring (G4) + docs

**Files:**
- Modify: `eval/experiment.py`, `eval/ragas_adapter.py`
- Modify: `docs/architecture.md`, `docs/PROJECT_STATUS.md`
- Test: `tests/test_sp12_eval_wiring.py`

**Interfaces:**
- Consumes: `build(enable_rewriter=...)` from Task 4
- Produces: eval `build()` calls pass `enable_rewriter=True`

- [ ] **Step 1: Write the failing test** — `tests/test_sp12_eval_wiring.py`
```python
import inspect

import eval.experiment as experiment
import eval.ragas_adapter as ragas_adapter


def test_eval_enables_rewriter_g4():
    # Both eval entry points must build the pipeline with the rewriter ON so the
    # gate measures recall impact (spec G4). Guard against silent regression.
    for mod in (experiment, ragas_adapter):
        src = inspect.getsource(mod)
        assert "enable_rewriter=True" in src, f"{mod.__name__} must enable the rewriter (G4)"
```

- [ ] **Step 2: Run to verify it fails**
Run: `.venv/bin/python -m pytest tests/test_sp12_eval_wiring.py -v`
Expected: FAIL (assertion).

- [ ] **Step 3: Implement** — add `enable_rewriter=True` to the `build(...)` call in `eval/experiment.py` (the call currently passing `enable_guardrails=False, enable_cache=False`) and in `eval/ragas_adapter.py` (same). Update `docs/architecture.md` and `docs/PROJECT_STATUS.md`: add SP12 to the query-path diagram/description and the config-knob table (`rewriter_enabled` / `rewriter_llm_enabled` / `rewriter_llm_threshold`, defaults True/True/5, synonym key format `rewriter:synonyms:{tenant_id}`, note "retrieval-only; generation stays on the original question; enabled in eval per G4").

- [ ] **Step 4: Run to verify it passes**
Run: `.venv/bin/python -m pytest tests/test_sp12_eval_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add eval/experiment.py eval/ragas_adapter.py docs/architecture.md docs/PROJECT_STATUS.md tests/test_sp12_eval_wiring.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): enable rewriting in eval runs (G4) + docs (SP12)"
```

---

## Final verification
- [ ] Full offline suite: `.venv/bin/python -m pytest -q` — green.
- [ ] Lint: `.venv/bin/python -m ruff check core/ providers/rewriter/ eval/ tests/test_sp12_*.py` — 0 errors.
- [ ] Import-isolation sanity: `.venv/bin/python -c "import core.pipeline, providers.rewriter.hybrid_rewriter"` succeeds with no Redis running (offline-safe).
