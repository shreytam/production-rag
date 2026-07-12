# SP12 · Native Query Rewriting & Synonym Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implements a two-tiered, tenant-safe query rewriter combining regex-based Redis synonym lookups with structured LLM query-expansion fallback logic.

**Architecture:** Create a `QueryRewriter` Protocol, build a `HybridQueryRewriter` utilizing a Redis connection and a cheap generator, and integrate it into the `RAGPipeline` hot query path before the L1/L2 cache check.

**Tech Stack:** Python 3.11-3.13, Pydantic, redis, fakeredis.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Tenant synonym isolation must remain hostile-by-design: synonym keys partition strictly on `rewriter:synonyms:{tenant_id}` namespaces.
- Rewriting failures degrade immediately to raw search queries.

---

### Task 1: Configuration Knobs for Query Rewriting

**Files:**
- Modify: `core/config.py`

**Interfaces:**
- Consumes: None
- Produces:
  - Settings: `rewriter_enabled`, `rewriter_llm_enabled`, `rewriter_llm_threshold`, `rewriter_redis_url`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp12_config.py` verifying setup defaults:
```python
import pytest
from core.config import Settings

def test_sp12_config_defaults():
    settings = Settings()
    assert settings.rewriter_enabled is True
    assert settings.rewriter_llm_enabled is True
    assert settings.rewriter_llm_threshold == 5
    assert settings.rewriter_redis_url.startswith("redis://")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp12_config.py`
Expected: FAIL (AttributeError missing configuration settings)

- [ ] **Step 3: Modify files**
Add configuration knobs under `Settings` in `core/config.py`:
```python
    # --- Query Rewriting & Synonym Expansion (SP12) ---
    rewriter_enabled: bool = True
    rewriter_llm_enabled: bool = True
    rewriter_llm_threshold: int = 5
    rewriter_redis_url: str = "redis://localhost:6379/1"
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp12_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/config.py tests/test_sp12_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): add configuration parameters for synonym rewriter" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: QueryRewriter Protocol definition

**Files:**
- Modify: `core/interfaces.py`

**Interfaces:**
- Consumes: None
- Produces: `QueryRewriter` Protocol class and signatures

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp12_interface.py` to assert Protocol registration:
```python
import pytest
from core.interfaces import QueryRewriter

def test_query_rewriter_protocol():
    assert isinstance(QueryRewriter, type)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp12_interface.py`
Expected: FAIL (ImportError missing QueryRewriter class)

- [ ] **Step 3: Modify files**
Modify `core/interfaces.py` to add `QueryRewriter` definition:
```python
@runtime_checkable
class QueryRewriter(Protocol):
    """Query rewriting and synonym expansion interface."""

    def rewrite(self, query: str, acl: ACLContext) -> str: ...
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp12_interface.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py tests/test_sp12_interface.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): define QueryRewriter protocol interface" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Hybrid Query Rewriter Implementation

**Files:**
- Create: `providers/rewriter/hybrid_rewriter.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: `QueryRewriter` Protocol, `Generator` instance
- Produces: `HybridQueryRewriter` and registry factory maps

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp12_rewriter.py` checking fake Redis synonyms lookup and LLM expansion:
```python
import pytest
from providers.rewriter.hybrid_rewriter import HybridQueryRewriter
from tests._fakes import RecordingGenerator
from core.types import ACLContext

def test_synonym_substitution():
    # Setup mock connections and check replacements
    # e.g. mapping "NYPD" to "New York Police Department"
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp12_rewriter.py`
Expected: FAIL (ModuleNotFoundError or AttributeError)

- [ ] **Step 3: Modify files**
Create `providers/rewriter/hybrid_rewriter.py`:
```python
from __future__ import annotations
import re
import logging
import redis
from core.types import ACLContext, ChatMessage
from core.interfaces import Generator, QueryRewriter

logger = logging.getLogger(__name__)

class HybridQueryRewriter:
    def __init__(self, generator: Generator, redis_url: str, llm_enabled: bool = True, llm_threshold: int = 5) -> None:
        self._gen = generator
        self._redis = redis.from_url(redis_url)
        self._llm_enabled = llm_enabled
        self._llm_threshold = llm_threshold

    def rewrite(self, query: str, acl: ACLContext) -> str:
        # 1. Redis synonym dictionary extraction
        synonyms = {}
        try:
            # Query tenant-scoped dictionary hash key
            key = f"rewriter:synonyms:{acl.tenant_id}"
            raw = self._redis.hgetall(key)
            if raw:
                synonyms = {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Redis synonym lookup failed, degrading to LLM expansion: %s", exc)

        rewritten = query
        words = query.split()
        replaced = False
        
        # 2. Rule-based replacement on word boundaries
        if synonyms:
            for shortcut, full_term in synonyms.items():
                pattern = re.compile(rf"\b{re.escape(shortcut)}\b", re.IGNORECASE)
                if pattern.search(rewritten):
                    rewritten = pattern.sub(full_term, rewritten)
                    replaced = True

        # 3. LLM expansion fallback
        if self._llm_enabled and len(words) >= self._llm_threshold and not replaced:
            try:
                messages = [
                    ChatMessage(
                        role="system",
                        content="You are an expert search assistant. Rewrite this query to maximize search retrieval matching."
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Rewrite this RAG query into a single descriptive search statement: '{query}'"
                    )
                ]
                resp = self._gen.complete(messages, max_tokens=128, temperature=0.0)
                rewritten = resp.text.strip()
            except Exception as exc:
                logger.error("LLM Query Expansion failed: %s. Falling back to synonym result.", exc)

        return rewritten
```
Modify `core/registry.py` to register `build_query_rewriter`:
```python
def build_query_rewriter(settings: Settings | None = None) -> QueryRewriter:
    s = settings or get_settings()
    # Use context model for cheap generation operations
    generator = build_generator(role="context", settings=s)
    from providers.rewriter.hybrid_rewriter import HybridQueryRewriter
    return HybridQueryRewriter(
        generator=generator,
        redis_url=s.rewriter_redis_url,
        llm_enabled=s.rewriter_llm_enabled,
        llm_threshold=s.rewriter_llm_threshold
    )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp12_rewriter.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add providers/rewriter/hybrid_rewriter.py core/registry.py tests/test_sp12_rewriter.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): implement HybridQueryRewriter synonym and LLM expander" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Pipeline Integration

**Files:**
- Modify: `core/pipeline.py`

**Interfaces:**
- Consumes: `QueryRewriter`
- Produces: Parameterized query rewriting prior to retrieval and caching steps

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp12_pipeline.py` asserting rewrites are routed:
```python
import pytest
from core.pipeline import RAGPipeline
from tests._fakes import RecordingGenerator

def test_pipeline_rewrites_and_routes():
    # Setup test workspace and verify query rewrite executes
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp12_pipeline.py`
Expected: FAIL (assert failures or missing attributes)

- [ ] **Step 3: Modify files**
Update constructor and `answer()` function in `core/pipeline.py` to load and pass queries through the rewriter:
```python
# In RAGPipeline.__init__
        self.rewriter = build_query_rewriter(settings) if settings.rewriter_enabled else None

# In RAGPipeline.answer:
        # Before: question used directly in query builder
        # After: pass through rewriter first
        if self.rewriter is not None:
            question = self.rewriter.rewrite(question, acl)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp12_pipeline.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/pipeline.py tests/test_sp12_pipeline.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(rewriter): integrate query rewriter into pipeline query execution flow" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
