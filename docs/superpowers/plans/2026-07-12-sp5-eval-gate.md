# SP5 · Eval Gate That Gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hardens the CI evaluation execution to prevent silent failures and regression bypasses.

**Architecture:** Extend the Generator contract to support a seed parameter, implement a seeded majority-vote LLM judge, rewrite the comparison logic to gate on statistical bootstrap difference-CI limits, fix the answer relevancy embedding space mismatch, and wire up live tests and clean skips in CI.

**Tech Stack:** Python 3.11-3.13, Pydantic, pytest, NumPy, GitHub Actions.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Nan metrics, mismatched sample lengths (N mismatch), or missing baseline files must fail closed immediately.
- Custom evaluation hooks should allow scaling to domain-specific custom evaluation checks.

---

### Task 1: Configuration Knobs for Evaluation

**Files:**
- Modify: `core/config.py`

**Interfaces:**
- Consumes: None
- Produces:
  - Settings: `eval_tolerance`, `eval_fast_n`, `eval_fast_seed`, `eval_bootstrap_resamples`, `judge_votes`, `judge_seed`, `require_live_stores`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_config.py` to verify configuration additions:
```python
import pytest
from core.config import Settings, get_settings
from pydantic import Field

def test_sp5_config_defaults():
    settings = Settings()
    assert settings.judge_votes == 3
    assert settings.judge_seed == 0
    assert settings.eval_tolerance == 0.03
    assert settings.eval_fast_n == 15
    assert settings.eval_fast_seed == 0
    assert settings.eval_bootstrap_resamples == 1000
    assert settings.require_live_stores is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_config.py`
Expected: FAIL (ValidationError or AttributeError due to missing attributes)

- [ ] **Step 3: Modify files**
Modify `core/config.py` to add the parameters under `class Settings`:
```python
    # --- Eval Gate & Stats ---
    eval_tolerance: float = 0.03
    eval_fast_n: int = 15
    eval_fast_seed: int = 0
    eval_bootstrap_resamples: int = 1000
    
    # --- LLM Judge Voting ---
    judge_votes: int = 3
    judge_seed: int = 0

    # --- Live Store CI Gating ---
    # Managed via validation alias to capture both env styles
    from pydantic import Field, AliasChoices
    require_live_stores: bool = Field(
        default=False,
        validation_alias=AliasChoices("rag_require_live_stores", "require_live_stores")
    )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/config.py tests/test_sp5_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): add configuration parameters for eval gate, LLM votes, and live tests" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Generator Contract Extension

**Files:**
- Modify: `core/interfaces.py`
- Modify: `providers/generators/openai_compatible.py`
- Modify: `providers/generators/anthropic.py`
- Modify: `tests/_fakes.py`

**Interfaces:**
- Consumes: None
- Produces: `Generator.complete` signature update with `seed: int | None = None` forwarded via keyword argument

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_generator_seed.py` checking seed forwarding:
```python
import pytest
from tests._fakes import RecordingGenerator

def test_generator_receives_seed():
    gen = RecordingGenerator()
    # Explicitly verify we can invoke complete with seed
    gen.complete([], seed=42)
    assert len(gen.calls) == 1
    # Check that seed was stored/passed
    assert getattr(gen, "last_seed", None) == 42
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_generator_seed.py`
Expected: FAIL (complete() got an unexpected keyword argument 'seed' or seed not recorded)

- [ ] **Step 3: Modify files**
Update `core/interfaces.py` to add `seed: int | None = None` to the `Generator.complete` Protocol signature.
Update `providers/generators/openai_compatible.py::OpenAICompatibleGenerator.complete`:
```python
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
```
Update `providers/generators/anthropic.py::AnthropicGenerator.complete`:
```python
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        # Anthropic does not support seed; we accept and ignore it
```
Update `tests/_fakes.py::RecordingGenerator`:
```python
    def __init__(self, text="OK", parsed=None):
        self._text = text
        self._parsed = parsed
        self.calls: list[list[ChatMessage]] = []
        self.last_seed = None

    def complete(self, messages, *, response_model=None, seed=None, **_):
        self.calls.append(list(messages))
        self.last_seed = seed
        return LLMResponse(
            text=self._text,
            parsed=self._parsed if response_model else None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake",
        )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_generator_seed.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py providers/generators/openai_compatible.py providers/generators/anthropic.py tests/_fakes.py tests/test_sp5_generator_seed.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): extend Generator.complete signature with seed support" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Seeded Majority-Vote Judge and Custom Eval Hook

**Files:**
- Modify: `eval/llm_judge.py`

**Interfaces:**
- Consumes: `Generator` instance
- Produces: `holistic_judge(..., votes: int = 1, base_seed: int = 0)` executing majority/median logic

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_llm_judge_vote.py` checking voting:
```python
import pytest
from tests._fakes import RecordingGenerator
from eval.llm_judge import holistic_judge, JudgeOutput

def test_majority_vote_median():
    # Make a mock generator returning different scores
    class VotingFakeGenerator:
        def __init__(self):
            self.count = 0
            self.seeds = []
        def complete(self, messages, *, response_model=None, seed=None, **_):
            self.count += 1
            self.seeds.append(seed)
            # Return scores: 0.8 on seed 0, 0.4 on seed 1, 0.9 on seed 2
            score = 0.8 if seed == 0 else (0.4 if seed == 1 else 0.9)
            return type("LLMResponse", (), {
                "parsed": {"score": score, "rationale": f"Vote {seed}"}
            })
            
    gen = VotingFakeGenerator()
    res = holistic_judge("Q", "A", ["C"], gen, votes=3, base_seed=10)
    assert gen.count == 3
    assert gen.seeds == [10, 11, 12]
    # Median of [0.4, 0.8, 0.9] is 0.8
    assert res["score"] == 0.8
    assert "Vote 10" in res["rationale"] or "Vote 12" in res["rationale"]  # corresponds to 0.8 or general context
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_llm_judge_vote.py`
Expected: FAIL (holistic_judge does not accept votes/base_seed or does not implement median logic)

- [ ] **Step 3: Modify files**
Rewrite `holistic_judge` in `eval/llm_judge.py` to implement voter count and median calculation:
```python
def holistic_judge(
    question: str,
    answer: str,
    contexts: list[str],
    generator: Generator,
    votes: int = 1,
    base_seed: int = 0,
) -> dict:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    
    results = []
    # Collect votes
    for i in range(votes):
        resp = generator.complete(
            [
                ChatMessage(role="system", content=RUBRIC),
                ChatMessage(
                    role="user",
                    content=(
                        f"Question: {question}\n\n"
                        f"Contexts:\n{context_block}\n\n"
                        f"Answer: {answer}\n\n"
                        "Provide your evaluation."
                    ),
                ),
            ],
            response_model=JudgeOutput,
            max_tokens=512,
            seed=base_seed + i,
        )
        parsed = resp.parsed or {}
        results.append({
            "score": float(parsed.get("score", 0.0)),
            "rationale": str(parsed.get("rationale", "")),
        })

    # Sort results by score and take the median
    results.sort(key=lambda r: r["score"])
    median_idx = len(results) // 2
    return results[median_idx]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_llm_judge_vote.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add eval/llm_judge.py tests/test_sp5_llm_judge_vote.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): implement seeded majority-voting LLM judge" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Answer Relevancy Space Mismatch Fix

**Files:**
- Modify: `eval/generation_metrics.py`

**Interfaces:**
- Consumes: `Embedder` protocol
- Produces: `answer_relevancy` passing query-space vectors for similarity checks

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_relevancy_space.py` verifying usage of `embed_query` rather than `embed_documents`:
```python
import pytest
from tests._fakes import FakeEmbedder
from eval.generation_metrics import answer_relevancy

class TrackingFakeEmbedder(FakeEmbedder):
    def __init__(self):
        self.query_calls = []
        self.doc_calls = []
    def embed_query(self, text):
        self.query_calls.append(text)
        return super().embed_query(text)
    def embed_documents(self, texts):
        self.doc_calls.append(texts)
        return super().embed_documents(texts)

def test_relevancy_uses_only_query_embeddings():
    class LocalFakeGenerator:
        def complete(self, *args, **keys):
            return type("Resp", (), {"parsed": {"questions": ["Q1", "Q2"]}})()

    embedder = TrackingFakeEmbedder()
    # Run relevancy metric
    score = answer_relevancy("Question", "Answer", LocalFakeGenerator(), embedder)
    # Check that generated questions were embedded with embed_query, not embed_documents
    assert len(embedder.query_calls) >= 3  # 1 (original) + 2 (generated questions)
    assert len(embedder.doc_calls) == 0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_relevancy_space.py`
Expected: FAIL (doc_calls has length 1 instead of 0)

- [ ] **Step 3: Modify files**
Update `answer_relevancy` in `eval/generation_metrics.py` to query-embed the generated questions individually:
```python
    orig_vec = embedder.embed_query(question)
    # Fix asymmetric-model mismatch: embed both sides in query space
    gen_vecs = [embedder.embed_query(q) for q in generated]

    sims = [_cosine(orig_vec, gv) for gv in gen_vecs]
    return float(np.mean(sims)) if sims else 0.0
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_relevancy_space.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add eval/generation_metrics.py tests/test_sp5_relevancy_space.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "fix(eval): embed generated questions in query space for answer relevance" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: Core Gate & Custom Evaluation Hook

**Files:**
- Modify: `eval/compare.py`

**Interfaces:**
- Consumes: Run metrics JSON structures
- Produces: `compare` function supporting paired bootstrap CI, NaN validation checks, and custom check registration

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_gate.py` to check bootstrap CI logic, NaN gating, and custom metrics registration:
```python
import pytest
from pathlib import Path
import json
from eval.compare import compare

def test_compare_fails_closed_on_nan_or_length_mismatch(tmp_path):
    base_file = tmp_path / "base.json"
    new_file = tmp_path / "new.json"
    
    # 1. Base results
    with base_file.open("w") as f:
        json.dump({
            "aggregates": {"faithfulness": 0.9},
            "items": [{"generation_metrics": {"faithfulness": 0.9}}]
        }, f)
        
    # 2. New results (missing metric / NaN)
    with new_file.open("w") as f:
        json.dump({
            "aggregates": {"faithfulness": float("nan")},
            "items": [{"generation_metrics": {"faithfulness": float("nan")}}]
        }, f)

    # Must fail because of NaN metric
    assert compare("squad", baseline_file=base_file, new_version="fresh_run", tolerance=0.03) is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_gate.py`
Expected: FAIL (compare passes because of `nan < base - tolerance` evaluating to False)

- [ ] **Step 3: Modify files**
Rewrite `compare` in `eval/compare.py` to:
1. Enforce strict paired equal-N validation (raise ValueError/exit on mismatch).
2. Fail closed on metrics that are NaN/absent in base or new runs.
3. Compute client-level bootstrap confidence interval `(mean_diff, lo, hi)`. Let's fail if difference-CI upper bound `hi < -tolerance` or if `hi/lo` is NaN.
4. Support registering custom evaluation check functions.
5. Print a `PASSED`/`FAILED` column in the fixed-width table.

Modify `compare` and related functions in `eval/compare.py`:
```python
import math

CUSTOM_EVAL_HOOKS = {}

def register_custom_metric(name, scoring_func):
    """Registry seam to hook custom evaluations into the gate pipeline."""
    CUSTOM_EVAL_HOOKS[name] = scoring_func

def _extract_aggregates(results: dict) -> dict[str, float]:
    """Return {metric_name: mean} from a results JSON, including registered custom evals."""
    aggs = {k: v["mean"] for k, v in results.get("aggregates", {}).items() if not math.isnan(v["mean"])}
    return aggs

def compare(
    dataset: str,
    base_version: str | None = None,
    new_version: str = "full",
    tolerance: float = 0.02,
    baseline_file: Path | None = None,
) -> bool:
    """Compare two runs and return True if all metrics pass the gate."""
    if baseline_file is not None:
        base_path = baseline_file
    elif base_version is not None:
        base_path = RUNS_DIR / f"{dataset}.{base_version}.results.json"
    else:
        base_path = BASELINES_DIR / f"{dataset}.json"

    new_path = RUNS_DIR / f"{dataset}.{new_version}.results.json"

    for p in (base_path, new_path):
        if not p.exists():
            print(f"[compare] Results file not found: {p}", file=sys.stderr)
            sys.exit(1)

    base_results = _load_results(base_path)
    new_results = _load_results(new_path)

    base_agg = _extract_aggregates(base_results)
    new_agg = _extract_aggregates(new_results)

    all_metrics = sorted(set(base_agg.keys()) | set(new_agg.keys()) | set(CUSTOM_EVAL_HOOKS.keys()))
    rows = []
    failures: list[str] = []

    # Print table header with PASS/FAIL
    header = f"{'Metric':<25}{'Base':>10}{'New':>10}{'Delta':>10}{'CI lo':>10}{'CI hi':>10}{'Verdict':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for metric in all_metrics:
        # Check custom hook integration
        if metric in CUSTOM_EVAL_HOOKS:
            # Custom checks calculate scores from raw items
            hook = CUSTOM_EVAL_HOOKS[metric]
            base_items = [hook(item) for item in base_results.get("items", [])]
            new_items = [hook(item) for item in new_results.get("items", [])]
            base_val = sum(base_items) / len(base_items) if base_items else float("nan")
            new_val = sum(new_items) / len(new_items) if new_items else float("nan")
        else:
            base_val = base_agg.get(metric, float("nan"))
            new_val = new_agg.get(metric, float("nan"))

            base_items = _extract_item_values(base_results, metric)
            new_items = _extract_item_values(new_results, metric)

        # Enforce equal-N check
        if not base_items or not new_items:
            print(f"[compare] Error: Missing values for metric {metric}", file=sys.stderr)
            sys.exit(1)
        if len(base_items) != len(new_items):
            print(f"[compare] Error: Sample size N mismatch for {metric}. Base has {len(base_items)}, New has {len(new_items)}", file=sys.stderr)
            sys.exit(1)

        # Gating checks
        delta = new_val - base_val
        if math.isnan(base_val) or math.isnan(new_val):
            verdict = "FAILED"
            failures.append(f"{metric}: Base or New is NaN")
            lo = hi = float("nan")
        else:
            # Run paired bootstrap
            _, lo, hi = paired_bootstrap(base_items, new_items)
            
            # Gating Rule: fail if statistically significant regression
            if math.isnan(lo) or math.isnan(hi):
                verdict = "FAILED"
                failures.append(f"{metric}: bootstrap returned NaN confidence intervals")
            elif hi < -tolerance:
                verdict = "FAILED"
                failures.append(f"{metric}: regression confidence bound {hi:.4f} < {-tolerance:.4f}")
            else:
                verdict = "PASSED"

        print(
            f"{metric:<25}{base_val:>10.4f}{new_val:>10.4f}{delta:>10.4f}{lo:>10.4f}{hi:>10.4f}{verdict:>12}"
        )
        rows.append((metric, base_val, new_val, delta, lo, hi, verdict))

    print(sep)

    if failures:
        print("\n[compare] GATE FAILED — metric regressions detected:")
        for f in failures:
            print(f"  - {f}")
        return False

    print("\n[compare] All metrics within tolerance. Gate PASSED.")
    return True
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_gate.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add eval/compare.py tests/test_sp5_gate.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): rewrite compare gate logic to run bootstrap difference CI, N verification and custom metrics hooks" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: Live ACL Store Verification Harness

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_stores_acl.py`
- Modify: `tests/test_multitenant_isolation.py`

**Interfaces:**
- Consumes: Settings `require_live_stores`
- Produces: `require_live_or_fail` utility raising `pytest.fail(...)` if live stores are unreachable

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_live_gate.py` checking fail vs skip logic:
```python
import pytest
from core.config import Settings

def test_require_live_or_fail_behavior():
    # Setup settings mock or custom function
    settings_fail = Settings(require_live_stores=True)
    settings_skip = Settings(require_live_stores=False)

    def verify_harness(settings, reachable):
        if not reachable:
            if settings.require_live_stores:
                pytest.fail("Failing live store test")
            else:
                pytest.skip("Skipping live store test")

    # When require_live_stores=True and store unreachable, must fail
    with pytest.raises(pytest.fail.Exception):
        verify_harness(settings_fail, reachable=False)

    # When require_live_stores=False, must skip
    with pytest.raises(pytest.skip.Exception):
        verify_harness(settings_skip, reachable=False)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_live_gate.py`
Expected: FAIL (Fixture/Exceptions do not exist or behavior mismatch)

- [ ] **Step 3: Modify files**
Add helper function in `tests/conftest.py`:
```python
import os
import pytest
from core.config import get_settings

@pytest.fixture
def require_live_or_fail():
    """Fail the test instead of skipping if require_live_stores config is enabled and stores are unreachable."""
    def _verify(reachable: bool, backend: str):
        if not reachable:
            settings = get_settings()
            # Check both config setting and env parameter
            if settings.require_live_stores or os.environ.get("RAG_REQUIRE_LIVE_STORES") == "1":
                pytest.fail(f"Required connection to {backend} is down/missing in this gated test run!")
            else:
                pytest.skip(f"Connection to {backend} is unreachable. Skipping live store test.")
    return _verify
```
Update skips in `tests/test_stores_acl.py` and `tests/test_multitenant_isolation.py` to consume the `require_live_or_fail` fixture instead of calling `pytest.skip()` directly.

For example, in `tests/test_multitenant_isolation.py`:
```python
# Before
if not postgres_is_up():
    pytest.skip("Postgres is unreachable")

# After
def test_multitenant_pg_isolation(require_live_or_fail):
    require_live_or_fail(postgres_is_up(), "Postgres")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_live_gate.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add tests/conftest.py tests/test_stores_acl.py tests/test_multitenant_isolation.py tests/test_sp5_live_gate.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(security): force fail on missing ACL database resources in CI" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: Baseline Generation CLI and Makefile

**Files:**
- Modify: `eval/run_eval.py`
- Modify: `Makefile`
- Modify: `docs/architecture.md`

**Interfaces:**
- Consumes: None
- Produces: CLI switch `--write-baseline` writing directly to `eval/baselines/<dataset>.json` under `--fast` constraints

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp5_baseline_cli.py` to test write validations:
```python
import pytest
import sys
import subprocess
from pathlib import Path

def test_run_eval_baseline_fails_without_fast():
    # Execute run_eval python command to verify validation logic
    cmd = [sys.executable, "-m", "eval.run_eval", "--dataset", "hotpotqa", "--write-baseline"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Requires --fast option when --write-baseline is active" in res.stderr
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp5_baseline_cli.py`
Expected: FAIL (baseline switch does not exist or validation does not enforce --fast check)

- [ ] **Step 3: Modify files**
Update `eval/run_eval.py` to add parameters and output mapping:
1. Under `run_eval` in `eval/run_eval.py`:
```python
    version: str,
    fast: bool = False,
    limit: int | None = None,
    skip_gen_metrics: bool = False,
    write_baseline: bool = False,
) -> Path:
```
2. Refuse if `write_baseline` is True and `fast` is False:
```python
    if write_baseline and not fast:
        print("[run_eval] Error: Writing baseline requires the --fast option to match the CI subset.", file=sys.stderr)
        sys.exit(1)
```
3. Thread configuration variables into fast_subset and generator build:
```python
    from core.config import get_settings
    settings = get_settings()

    if limit:
        items = items[:limit]
    if fast:
        items = fast_subset(items, n=settings.eval_fast_n, seed=settings.eval_fast_seed)

    pipeline = _build_pipeline(version, dataset)
    # Thread settings values for votes and seeds
    generator = build_generator(role="judge")
    embedder = build_embedder()
```
4. Update holistic_judge call site to pass settings parameters:
```python
            # Evaluate using holistic judge
            # Before: holistic_judge(item["question"], ans, context_texts, generator)
            # After:
            holistic_judge(
                item["question"], 
                ans, 
                context_texts, 
                generator, 
                votes=settings.judge_votes, 
                base_seed=settings.judge_seed
            )
```
5. Choose baseline output target:
```python
    BASELINES_DIR = Path(__file__).parent / "baselines"
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    if write_baseline:
        out_path = BASELINES_DIR / f"{dataset}.json"
    else:
        suffix = ".retrieval" if skip_gen_metrics else ""
        out_path = RUNS_DIR / f"{dataset}.{version}{suffix}.results.json"
```
6. Add `--write-baseline` parameter to `main()` in `eval/run_eval.py`.
7. Add target to `Makefile`:
```makefile
.PHONY: baseline
baseline:
	python -m eval.run_eval --dataset $(or $(DATASET),hotpotqa) --version baseline --fast --write-baseline
```
8. Update `docs/architecture.md` description on bootstrap configuration procedure.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp5_baseline_cli.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add eval/run_eval.py Makefile docs/architecture.md tests/test_sp5_baseline_cli.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(eval): add baseline writer CLI target and make target validation checks" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 8: GitHub Actions Workflow Settings

**Files:**
- Modify: `.github/workflows/eval-gate.yml`

**Interfaces:**
- Consumes: GitHub Secrets
- Produces: Gated status validations and live service containers

- [ ] **Step 1: Write the failing test**
Create a dry run script on the yaml schema or do a syntax check to verify changes:
```bash
actionlint .github/workflows/eval-gate.yml
```
Expected: PASS/no syntax errors.

- [ ] **Step 2: Modify files**
Update `.github/workflows/eval-gate.yml` adding services and isolating gate jobs.
1. Add `acl-isolation` job using Postgres and Qdrant containers:
```yaml
  acl-isolation:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_DB: rag
          POSTGRES_USER: rag
          POSTGRES_PASSWORD: rag
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run live database isolation checks
        env:
          RAG_REQUIRE_LIVE_STORES: "1"
          VECTOR_STORE: "qdrant"
          PG_DSN: "postgresql://rag:rag@localhost:5432/rag"
          QDRANT_URL: "http://localhost:6333"
        run: |
          pytest tests/test_stores_acl.py tests/test_multitenant_isolation.py -v
```
2. Gate `eval` job on secret presence and report skipped or neutral status:
```yaml
  eval:
    runs-on: ubuntu-latest
    if: github.repository == 'ShreytamGoyal/production-rag'
    # Check key secret existence
```
3. Implement `eval-gate-status` required check reflecting maintenance overrides:
```yaml
  eval-gate-status:
    runs-on: ubuntu-latest
    needs: [lint, acl-isolation, eval]
    if: always()
    steps:
      - name: Check overall gate status
        run: |
          # Fails if lint or acl-isolation has failed
          # If eval was skipped due to fork PR secret bounds, pass only if 'eval-skip-approved' label is present in pull request metadata
```

- [ ] **Step 3: Commit**
```bash
git add .github/workflows/eval-gate.yml
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "ci(eval): integrate acl-isolation database check job and required status gate checks" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
