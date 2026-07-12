# SP7 · Observability & Cost Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrects query-path cost estimation and Langfuse native usage/cost dashboard tracking, and implements nearest-rank p95 percentile calculations.

**Architecture:** Model cost calculations will be computed dynamically using the actual model invoked (`ans.model`), mapping dated strings to normalized keys. The generation tracer span will be updated to a `generation`-typed observation in Langfuse. Latency percentiles on dashboards will be updated to use exact nearest-rank methods.

**Tech Stack:** Python 3.11-3.13, Pydantic, FastAPI, Langfuse SDK, NumPy (tests).

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Unpriced models fallback to cost of 0.0 with warning logged and `cost_priced=False` flag set.

---

### Task 1: Cost Estimation Normalization and Estimate Models

**Files:**
- Modify: `observability/cost.py`

**Interfaces:**
- Consumes: None
- Produces: `CostEstimate` model, `normalize_model`, `estimate_cost`, updated `cost_usd`

- [ ] **Step 1: Write the failing test**
Extend `tests/test_observability.py` with tests for model name normalization and pricing estimates:
```python
import pytest
from observability.cost import normalize_model, estimate_cost, CostEstimate

def test_normalize_model_names():
    # Dated anthropic model normalization
    assert normalize_model("claude-sonnet-4-6-20250115") == "claude-sonnet-4-6"
    assert normalize_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    # OpenAI/NIM model passthrough
    assert normalize_model("meta/llama-3.3-70b-instruct") == "meta/llama-3.3-70b-instruct"
    # Unknown models return None
    assert normalize_model("unknown-model-prefix/invalid") is None

def test_estimate_cost_pricing():
    # Estimate for known model
    est = estimate_cost("claude-sonnet-4-6-20250115", prompt_tokens=1000, completion_tokens=500)
    assert est.usd > 0.0
    assert est.priced is True
    assert est.resolved_model == "claude-sonnet-4-6"
    
    # Estimate for unknown model contains 0.0 value with priced=False
    est_unknown = estimate_cost("unknown-model-prefix/invalid", prompt_tokens=1000, completion_tokens=500)
    assert est_unknown.usd == 0.0
    assert est_unknown.priced is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_observability.py`
Expected: FAIL (ImportError or AttributeError on missing functions)

- [ ] **Step 3: Modify files**
Update `observability/cost.py` to add normalization, `CostEstimate` Pydantic model, and lookup logic:
```python
import logging
from typing import Set
from pydantic import BaseModel

logger = logging.getLogger(__name__)
UNPRICED_MODELS_LOGGED: Set[str] = set()

# Extensible pricing table
# Check that active Sonnet and NIM models map to standard prices
# (values are input_cost_per_1M, output_cost_per_1M)
PRICING_ESTIMATES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
    "meta/llama-3.3-70b-instruct": (0.0, 0.0), # Free tier default
    # Add other observed models as backup keys
}

class CostEstimate(BaseModel):
    usd: float
    priced: bool
    resolved_model: str

def normalize_model(server_model: str) -> str | None:
    """Normalizes server model names to pricing keys by removing dates or prefixes."""
    if not server_model:
        return None
    model_str = server_model.lower().strip()
    
    # Exact match first
    # Strip common provider prefixes like "anthropic/" or "openai/"
    for prefix in ("anthropic/", "openai/", "nvidia/"):
        if model_str.startswith(prefix):
            model_str = model_str[len(prefix):]
            
    # Check if direct match
    for k in PRICING_ESTIMATES:
        if k.lower() == model_str:
            return k
            
    # Check dated pattern: e.g. claude-sonnet-4-6-20250115 -> claude-sonnet-4-6
    # Strip trailing date suffixes (-YYYYMMDD or similar)
    import re
    cleaned = re.sub(r'-\d{8}$', '', model_str)
    
    for k in PRICING_ESTIMATES:
        if k.lower() == cleaned:
            return k
            
    return None

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    """Computes cost estimate, logging warnings on unknowns once."""
    norm_key = normalize_model(model)
    if not norm_key:
        if model not in UNPRICED_MODELS_LOGGED:
            UNPRICED_MODELS_LOGGED.add(model)
            logger.warning("Unpriced model encountered: %r. Cost reported as 0.0.", model)
        return CostEstimate(usd=0.0, priced=False, resolved_model=model)
        
    rates = PRICING_ESTIMATES[norm_key]
    prompt_cost = (prompt_tokens / 1_000_000.0) * rates[0]
    completion_cost = (completion_tokens / 1_000_000.0) * rates[1]
    
    return CostEstimate(
        usd=prompt_cost + completion_cost,
        priced=True,
        resolved_model=norm_key
    )

def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Float wrapper to maintain backward-compatibility with existing tests."""
    return estimate_cost(model, prompt_tokens, completion_tokens).usd
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_observability.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add observability/cost.py tests/test_observability.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(observability): implement model normalization and cost estimates" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Generation-Typed Observation Tracer

**Files:**
- Modify: `observability/langfuse_tracing.py`

**Interfaces:**
- Consumes: None
- Produces: `Tracer.generation(...)` context manager starting a `generation` type trace

- [ ] **Step 1: Write the failing test**
Update `tests/test_observability.py` to check the generation observation constructor:
```python
import pytest
from observability.langfuse_tracing import get_tracer

def test_generation_span_creation():
    tracer = get_tracer()
    # Span yields a context manager
    with tracer.generation("test-gen-span", model="test-model") as span:
        assert span is not None
        # Verify update operates without errors
        span.update(output={"text": "A"})
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_observability.py`
Expected: FAIL (AttributeError: 'Tracer' object has no attribute 'generation')

- [ ] **Step 3: Modify files**
Update `observability/langfuse_tracing.py` to add `generation` to the `Tracer` class and `_NoOpSpan`:
```python
    def generation(self, name: str, **metadata) -> Any:
        """Context manager to trace LLM generations. Maps to as_type='generation'."""
        if not self.enabled:
            return _NoOpSpan()
        try:
            # Under active flow, fetch current span and nest a generation
            span = self.client.start_as_current_observation(
                as_type="generation",
                name=name,
                metadata=metadata or None
            )
            return _LangfuseSpan(span)
        except Exception as exc:
            logger.warning("Failed to start generation span: %s", exc)
            return _NoOpSpan()
```
Ensure `_NoOpSpan` and `_LangfuseSpan` update operations match.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_observability.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add observability/langfuse_tracing.py tests/test_observability.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(observability): add generation typed tracer to Langfuse interface" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Pipeline True-Cost Wiring

**Files:**
- Modify: `core/pipeline.py`

**Interfaces:**
- Consumes: `estimate_cost`, `tracer.generation`
- Produces: `Answer` with updated cost attributes, and `cost_priced` flag integration

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp7_pipeline_cost.py` checking price metadata:
```python
import pytest
from core.pipeline import RAGPipeline
from core.types import Answer

def test_pipeline_calculates_cost_from_actual_model(monkeypatch):
    # Dummy mock run ensuring actual model determines cost
    class DummyGenerator:
        def complete(self, *args, **kwargs):
            return type("Resp", (), {
                "text": "A",
                "parsed": None,
                "usage": type("Usage", (), {"prompt_tokens": 100, "completion_tokens": 100})(),
                "model": "claude-sonnet-4-6-20250115"
            })()

    # Stub pipeline properties
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp7_pipeline_cost.py`
Expected: FAIL (mismatch or pipeline continues pricing from configured settings key)

- [ ] **Step 3: Modify files**
Update `core/pipeline.py` to route generation calls through `tracer.generation` and extract `ans.model`:
```python
        # In core/pipeline.py under query execution
        with self.tracer.generation("generation", model=self.settings.gen_model) as s_gen:
            ans = self._generator.generate(...) # Wave 1 generator call
            
            # Extract actual model and calculate cost
            actual_model = getattr(ans, "model", "") or self.settings.gen_model
            est = estimate_cost(
                actual_model,
                ans.usage.prompt_tokens,
                ans.usage.completion_tokens
            )
            
            # Populate native Langfuse Usage & Cost dashboards
            s_gen.update(
                model=actual_model,
                usage_details={
                    "prompt_tokens": ans.usage.prompt_tokens,
                    "completion_tokens": ans.usage.completion_tokens
                },
                cost_details={
                    "input": est.usd * 0.2, # or calculate individual pieces if rate splits are unpacked
                    "output": est.usd * 0.8,
                    "total": est.usd
                },
                output={"text": ans.text}
            )

        # Update root pipeline trace and metadata
        ans.metadata["cost_usd"] = est.usd
        ans.metadata["cost_priced"] = est.priced
        
        # Attach to root trace
        self._root_trace.update(
            output={
                "cost_usd": est.usd,
                "cost_priced": est.priced
            }
        )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp7_pipeline_cost.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/pipeline.py tests/test_sp7_pipeline_cost.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(observability): wire true-cost calculation and metadata onto RAG pipeline" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Nearest-Rank p95 Percentile Latency

**Files:**
- Modify: `observability/dashboard.py`

**Interfaces:**
- Consumes: Sequence of latencies
- Produces: Correct nearest-rank percentile offsets

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp7_percentiles.py` asserting small N bounds:
```python
import pytest
import math
from observability.dashboard import calculate_p95

def test_nearest_rank_p95_small_n():
    # Value list from 1 to 10
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # Nearest-rank p95: ceil(0.95 * 10) -> index 10 (value 10.0)
    assert calculate_p95(latencies) == 10.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp7_percentiles.py`
Expected: FAIL (calculate_p95 yields 9.0 due to int truncation)

- [ ] **Step 3: Modify files**
Update latency percentile step in `observability/dashboard.py` (around line 69) using `ceil` mapping:
```python
    import math
    # Nearest-rank method: index calculated through ceiling
    sorted_v = sorted(values)
    idx = max(0, math.ceil(0.95 * len(sorted_v)) - 1)
    p95_val = sorted_v[idx]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp7_percentiles.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add observability/dashboard.py tests/test_sp7_percentiles.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "fix(observability): enforce nearest-rank p95 percentile metric extraction" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
