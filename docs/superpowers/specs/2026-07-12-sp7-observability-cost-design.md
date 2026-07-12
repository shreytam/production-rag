# SP7 · Observability & Cost Correctness — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 1. Observability slice: makes every trace, metric, and dashboard report the *true* cost and model of each query. Depends on the `Answer.model` field (already populated by `GroundedGenerator`, Wave 1) and the `Tracer` facade (SP-observability, Wave 1). No dependency on SP1/SP2; SP3 (PII/sampling) and SP6 (resilience) own adjacent concerns kept out of scope in §3.

---

## 1. Context & problem

Every defect below was verified by reading current source (line numbers confirmed at spec time; they may drift as the file changes).

| # | Defect | Location | Reality |
|---|--------|----------|---------|
| 1 | Cost billed against the wrong model | `core/pipeline.py:160-164` | `cost = cost_usd(self.settings.gen_model, …)`. `gen_model` defaults to `"meta/llama-3.3-70b-instruct"` which is priced **$0.00** in `PRICING`. When `gen_provider=="anthropic"` the call actually goes to `settings.anthropic_model` (`"claude-sonnet-4-6"`, a *paid* model) — but cost is still computed from `gen_model`, so every paid Claude query is billed **$0.00** across the root trace, `ans.metadata["cost_usd"]`, and all downstream metrics. The truth is on `ans.model` (server-returned model id), which `app/demo.py:89-91` already uses — the pipeline ignores it. |
| 2 | Unknown models silently cost $0 | `observability/cost.py:30-31` | `if model not in PRICING: return 0.0`. A model string the server returns that isn't a `PRICING` key (e.g. a dated variant, a new alias, a typo) yields silent **$0.00** — indistinguishable from a genuine free-tier model. No log, no flag, no metric. Cost under-reporting is invisible. |
| 3 | `PRICING` keys don't match server-returned strings | `observability/cost.py:14-22` vs `providers/generators/*` | `LLMResponse.model` is set from `response.model` (the string the *server* returns) in both `providers/generators/anthropic.py:99` and `providers/generators/openai_compatible.py:129`. Anthropic returns dated ids (e.g. `claude-sonnet-4-6-YYYYMMDD`) that won't equal the un-dated `PRICING` key `"claude-sonnet-4-6"`. Result: even after fixing defect #1, cost silently falls through to defect #2 and returns $0. |
| 4 | Native Langfuse usage/cost panels are empty | `observability/langfuse_tracing.py:162-164`, `core/pipeline.py:126-135` | The generation stage is opened with `as_type="span"` and tokens are stuffed into free-form `metadata`. Langfuse populates its native **Usage** and **Cost** dashboards only from a **generation-typed** observation carrying `usage_details` / `cost_details` / `model`. A `span` with tokens in `metadata` leaves those panels blank; cost is un-aggregatable in the Langfuse UI. |
| 5 | p95 percentile math is wrong (biased low) | `observability/dashboard.py:69` | `p95_idx = max(0, int(len(sorted_v) * 0.95) - 1)`. The `int()` truncation *plus* the unconditional `-1` under-counts the rank. For n=10 it returns `sorted[8]` — the **90th**-percentile value, not p95 (nearest-rank p95 of 1..10 is 10). The reported p95 latency is systematically optimistic, worst at small sample sizes typical of eval runs. |

Net effect: the system reports **$0.00** for paid production traffic, hides unpriced models, shows empty native cost panels in Langfuse, and under-reports tail latency. SP7 makes cost and observability *correct*.

## 2. Goals

- Cost is computed from the **model actually invoked** (`ans.model`), not a hardcoded config default — everywhere cost is recorded (root trace, `ans.metadata`, generation span, eval records).
- An **unknown / unpriced** model is **loud**: logged at WARNING and flagged on the trace/metric, never silently $0.
- `PRICING` lookup **tolerates server-returned model strings** (dated Anthropic ids, provider prefixes) via a documented normalization, so a real paid call is never mis-priced as free.
- Langfuse **native Usage/Cost panels populate**: the generation stage is a `generation`-typed observation carrying `model` + `usage_details` + `cost_details`.
- The offline dashboard reports a **correct nearest-rank p95** (and the rank convention is documented and tested).
- All of the above is **offline-testable** — no live LLM, no Langfuse server, no network — and the disabled-tracer/no-op path stays byte-for-byte silent.

## 3. Non-goals (deferred) — owner named

- **Langfuse PII masking of raw prompt/answer text and trace `sample_rate`** → **SP3 (Compliance & Privacy)**. SP7 attaches token/cost/model to the generation span; it does not decide what query text is masked or which traces are sampled.
- **Retry/timeout/circuit-breaker behavior of the generators** → **SP6 (Resilience)**. SP7 reads the `usage`/`model` a call returns; it does not change how the call is made or retried.
- **Multi-call cost aggregation** (contextual-prefix step, LLM-judge, RAGAS backing model) → **future hook (§11)**. SP7 corrects the *answer-generation* cost on the query path; roll-up of ancillary LLM calls is out of this slice.
- **Authoritative/live pricing feed** → out of program scope. `PRICING` stays a hand-maintained estimate table (its docstring already says "confirm with provider"); SP7 only fixes *which key is looked up* and *what happens on a miss*.
- **New dashboards / Grafana / metric exporters** → out of scope. SP7 fixes the existing offline text dashboard and the existing Langfuse spans only.

---

## 4. Decisions (PROPOSED) — confirm/override on review

Each row leads with the best-practice option.

| # | Decision | Choice (PROPOSED) | Rationale |
|---|----------|-------------------|-----------|
| D1 | Which model to cost | Cost from **`ans.model or settings.gen_model`** (prefer the server-returned model; fall back to config only when empty) | Bills the model actually called; the fallback covers the rare empty-`model` response without reintroducing the hardcode as the primary source. |
| D2 | Unknown-model handling | **Log WARNING + return 0.0 + signal "unpriced"** to the caller (via a structured result, see D3), never raise | Fail *observably*, not *closed-crashing*: a pricing gap must not 500 a query, but it must be visible in logs and on the trace. |
| D3 | Cost API shape | Add `estimate_cost(model, prompt, completion) -> CostEstimate` returning `(usd, priced: bool, resolved_model: str)`; keep `cost_usd(...)->float` as a thin back-compat wrapper | A bare `float` can't distinguish "genuinely $0 free-tier" from "unpriced miss". A small result object carries the `priced` flag the pipeline needs to flag traces — without breaking existing callers/tests that expect a float. |
| D4 | `PRICING` ↔ server-string reconciliation | **Normalize then exact-match, then dated-prefix match**: lowercase, strip a known provider prefix, and match a `PRICING` key that is a **date-suffix-stripped prefix** of the server string (e.g. `claude-sonnet-4-6-20250115` → `claude-sonnet-4-6`). No fuzzy/substring guessing. | Anthropic returns dated ids; NIM/OpenAI return the requested id. A conservative prefix rule fixes the real mismatch without risking a wrong-price collision. Add the exact dated keys we've observed as explicit entries too (belt-and-suspenders). |
| D5 | Langfuse generation span | Make the **generation** stage a `generation`-typed observation; attach `model=ans.model`, `usage_details={"prompt_tokens","completion_tokens"}`, `cost_details={"input","output","total"}` **after** generation returns. (Key shapes match the SDK's first-party OpenAI integration — `langfuse/openai.py:1202` uses `cost_details={"input","output","total"}`; the generic docstring's `{"total_cost":…}` is an accepted alias but the explicit-breakdown form is preferred for the native panels.) | This is the shape Langfuse's native Usage/Cost panels read. Attaching post-call means we have real token counts and the resolved model. |
| D6 | Tracer surface for generation | Add an explicit **`tracer.generation(name, **md)`** context manager (parallel to `tracer.span`) whose no-op path is identical to `span`'s | Keeps the "concrete Langfuse detail lives only in `langfuse_tracing.py`" boundary; the pipeline asks for a *generation* by intent, not by passing `as_type` strings around. |
| D7 | p95 algorithm | **Nearest-rank**: `idx = max(0, ceil(0.95 * n) - 1)` on the sorted list | Standard, dependency-free, monotonic, and correct at small n (the eval regime). No numpy. Convention documented in the docstring + pinned by tests. |
| D8 | Where the cost fix lives | In **`core/pipeline.py`** (compute) + **`observability/cost.py`** (resolve/flag) + **`observability/langfuse_tracing.py`** (generation span) — no new provider, no registry change | Cost is a cross-cutting concern of the existing query path, not a swappable component; it doesn't warrant a Protocol. Follows the codebase's "small, single-purpose units, wired where they're used" pattern. |
| D9 | Unpriced-model flag propagation | Set `ans.metadata["cost_priced"] = False` and `root.update(output={..., "cost_priced": False})` on a miss | Makes under-reporting queryable both offline (metadata) and in Langfuse (trace output), matching how the pipeline already surfaces `cost_usd`. |

---

## 5. Architecture & components

No new Protocol and no `core/registry.py` change: cost/observability is a cross-cutting concern of the single query path, not a swappable provider (D8). Changes are localized and behavioral.

### 5.1 `observability/cost.py` — resolution + observability

- **`normalize_model(server_model: str) -> str | None`** — lowercase, strip a known provider prefix, and resolve to a `PRICING` key by exact match then date-suffix-stripped-prefix match (D4). Returns the matched key, or `None` on no match. Pure, no I/O.
- **`CostEstimate`** (a small frozen `pydantic.BaseModel` or `NamedTuple`): `usd: float`, `priced: bool`, `resolved_model: str`.
- **`estimate_cost(model, prompt_tokens, completion_tokens) -> CostEstimate`** — resolves via `normalize_model`; on a **miss** logs `logger.warning("unpriced model %r — cost reported as $0", model)` **once per distinct model** (module-level `set` guard to avoid log spam) and returns `CostEstimate(0.0, priced=False, resolved_model=model)`. On a hit returns the real cost with `priced=True`.
- **`cost_usd(model, prompt, completion) -> float`** — retained as `return estimate_cost(...).usd`, so every existing test and caller (`app/demo.py:91`, `update_usage_cost`) keeps working unchanged.
- **`update_usage_cost(usage, model)`** — unchanged public behavior; now benefits from normalization for free (it calls `cost_usd`).
- `PRICING` gains the exact dated Anthropic keys we've observed (D4 belt-and-suspenders) *and* the normalization handles unseen dates.

### 5.2 `observability/langfuse_tracing.py` — generation-typed span

- Add **`Tracer.generation(name, **metadata)`** — a context manager mirroring `Tracer.span`, but the enabled branch calls `start_as_current_observation(as_type="generation", name=name, metadata=metadata or None)`. The disabled/failed-create branch yields `_NoOpSpan()`, identical to `span`.
- `_LangfuseSpan.update(**kwargs)` already forwards `**kwargs` to the underlying observation's `.update(...)`, and the v3/v4 generation observation accepts `model`, `usage_details`, `cost_details` on `.update()` — so **no new handle method is needed**; the pipeline calls `s_gen.update(model=…, usage_details=…, cost_details=…, output=…)`. (Verified against the SDK: `start_as_current_observation(as_type="generation", …)` exists and the generation object takes these fields.)
- No-op path unchanged: `_NoOpSpan.update(**kwargs)` swallows everything.

### 5.3 `core/pipeline.py` — wire true cost onto the generation span + trace

- Open the generation stage via `self.tracer.generation("generation", model=self.settings.gen_model)` (name/intent; the *real* model is attached after the call).
- After `ans` returns, compute `est = estimate_cost(ans.model or self.settings.gen_model, ans.usage.prompt_tokens, ans.usage.completion_tokens)`.
- Attach to the generation span:
  `s_gen.update(model=ans.model or self.settings.gen_model, usage_details={"prompt_tokens": p, "completion_tokens": c}, cost_details={"input": in_usd, "output": out_usd, "total": est.usd}, output={...})`.
- Replace the `cost = cost_usd(self.settings.gen_model, …)` block (lines 160-164) with `cost = est.usd`, and set `ans.metadata["cost_usd"] = cost`, `ans.metadata["cost_priced"] = est.priced`.
- Root trace `output` gains `"cost_priced": est.priced` alongside the existing `cost_usd` (D9).
- The **refused-input** early return (`_refused`, cost 0.0, priced trivially True) is unchanged — no generation happened.

### 5.4 `observability/dashboard.py` — correct p95

- Replace line 69 with nearest-rank (D7):
  ```python
  import math
  idx = max(0, math.ceil(0.95 * len(sorted_v)) - 1)
  ```
- Docstring gains one line naming the convention ("nearest-rank p95"). No other stat changes.

## 6. Data flow

```
answer(question)
  └─ tracer.span("rag.query")                      # root trace
       ├─ guardrail.input (unchanged)
       ├─ tracer.span("retrieval")  (unchanged)
       ├─ tracer.generation("generation")          # ← now generation-typed
       │     grounded.generate() → Answer{ text, usage, model=<server id> }
       │     est = estimate_cost(ans.model or gen_model, p, c)
       │        └─ normalize_model(ans.model) → PRICING key | None
       │             └─ None ⇒ WARNING once + priced=False + 0.0
       │     s_gen.update(model=ans.model,
       │                  usage_details={prompt_tokens,completion_tokens},
       │                  cost_details={input,output,total})   # populates native panels
       ├─ guardrail.output (unchanged)
       └─ root.update(output={ cost_usd=est.usd, cost_priced=est.priced, … })
  ans.metadata["cost_usd"]    = est.usd
  ans.metadata["cost_priced"] = est.priced
```

Offline dashboard reads `cost_usd`/`latency_ms` from eval `results.json` exactly as before; only the p95 computation changes.

## 7. Config knobs (core/config.py)

SP7 needs **no new required knobs** — the fix is correctness, not configurability. One optional knob is proposed to bound log noise; if the reviewer prefers zero new config, the module-level dedup `set` (§5.1) already prevents spam and this can be dropped.

| Knob | Default | Purpose |
|------|---------|---------|
| `cost_warn_on_unpriced` (PROPOSED, optional) | `True` | When `True`, an unpriced model logs a WARNING (once per distinct model) and sets `cost_priced=False`. Set `False` only for known-noisy dev loops; the `cost_priced` flag is still emitted regardless. |

Existing knobs SP7 relies on (unchanged): `gen_model`, `gen_provider`, `anthropic_model`, `langfuse_enabled`, `langfuse_*`.

## 8. Error handling — explicit, fail-observable (not fail-silent)

- **Unpriced model** (security/billing-correctness path): **fail observable, not silent.** Return `0.0` but log WARNING and set `cost_priced=False`. Rationale: a pricing gap must never crash a paid production query (that would be a self-inflicted outage), but silent $0 is exactly the bug SP7 exists to kill — so it is always logged and flagged. This is the deliberate stance for a *cost-estimate* path (contrast with SP2's fail-*closed* stance for *safety* guards).
- **Tracer never breaks the request:** `Tracer.generation` mirrors `Tracer.span` — a failed `start_as_current_observation` yields `_NoOpSpan()`; `_LangfuseSpan.update` swallows exceptions (existing `try/except … pass`). A Langfuse outage degrades observability, never the answer.
- **Empty / malformed `ans.model`:** falls back to `settings.gen_model` (D1). If *that* is also unpriced, defect-#2 handling applies (flag + WARN).
- **`normalize_model` on garbage input** (empty string, `None`): returns `None` → treated as unpriced; never raises.
- **Dashboard on degenerate input:** `n=0` already short-circuits to `{}`; `n=1` yields `ceil(0.95)-1 = 0` → the single value. No `IndexError`.

## 9. Testing (TDD) — concrete, offline

All tests run with **no network, no Langfuse server, no live LLM** (matching `tests/test_observability.py`'s existing contract). Write each test first; watch it fail against current code.

**cost.py**
1. `test_normalize_dated_anthropic` — `normalize_model("claude-sonnet-4-6-20250115")` → `"claude-sonnet-4-6"`.
2. `test_normalize_exact_passthrough` — `normalize_model("meta/llama-3.3-70b-instruct")` → same key.
3. `test_normalize_unknown_returns_none` — `normalize_model("mistral/foo")` → `None`.
4. `test_estimate_paid_anthropic_nonzero` — dated Claude id → `CostEstimate.usd > 0` and `priced is True`.
5. `test_estimate_unpriced_flags_and_warns` — unknown model → `usd == 0.0`, `priced is False`, and a WARNING is emitted (assert via `caplog`).
6. `test_estimate_unpriced_warns_once_per_model` — two calls, same unknown model → exactly one warning.
7. `test_cost_usd_backcompat` — `cost_usd(known, 1000, 500)` returns the same float as before (existing tests in `TestCostUsd` must still pass unchanged).

**pipeline.py** (with a fake generator returning `Answer(model="claude-sonnet-4-6-20250115", usage=…)` and a no-op tracer)
8. `test_cost_uses_answer_model_not_config` — `gen_model` is the free NIM default, generator returns a paid dated Claude id → `ans.metadata["cost_usd"] > 0`.
9. `test_cost_priced_true_on_known` — known model → `ans.metadata["cost_priced"] is True`.
10. `test_cost_priced_false_on_unknown` — generator returns an unpriced id → `cost_usd == 0.0` and `cost_priced is False`.
11. `test_empty_answer_model_falls_back_to_config` — `Answer(model="")` → costs from `settings.gen_model` (no crash).
12. `test_refused_input_cost_zero_priced` — blocked input → `cost_usd == 0.0`, no generation span opened.

**langfuse_tracing.py** (spy tracer / fake client capturing calls)
13. `test_generation_span_disabled_is_noop` — `langfuse_enabled=False` → `tracer.generation("g")` yields a `_NoOpSpan`; `.update(model=…, usage_details=…, cost_details=…)` is silent, no import of `langfuse`.
14. `test_generation_span_passes_generation_type` — with a fake client, `tracer.generation(...)` calls `start_as_current_observation(as_type="generation", …)` (assert on captured kwargs).

**dashboard.py**
15. `test_p95_nearest_rank_small_n` — latencies `1..10` → `p95 == 10.0` (current code returns 9.0; this test fails first).
16. `test_p95_n100` — `1..100` → `p95 == 95.0`.
17. `test_p95_single_value` — `[42.0]` → `p95 == 42.0` (no IndexError).
18. `test_p95_uniform` — all-equal list → that value.

## 10. Files

**Modify**
- `observability/cost.py` — add `normalize_model`, `CostEstimate`, `estimate_cost`; rewrite `cost_usd` as a wrapper; add observed dated keys + `logger`.
- `observability/langfuse_tracing.py` — add `Tracer.generation(...)` context manager.
- `core/pipeline.py` — generation stage → `tracer.generation`; compute cost from `ans.model or settings.gen_model` via `estimate_cost`; attach `model`/`usage_details`/`cost_details` on the generation span; set `cost_priced` on `ans.metadata` and root trace output.
- `observability/dashboard.py` — nearest-rank p95 (line 69) + docstring note.
- `core/config.py` — (optional, D7/§7) add `cost_warn_on_unpriced: bool = True`.
- `tests/test_observability.py` — add cost-resolution, generation-span, and p95 tests.
- `tests/test_pipeline_integration.py` — add the true-cost / `cost_priced` pipeline tests.

**Create**
- None. (No new provider, no new Protocol, no registry change — per D8.)

## 11. Open questions / future hooks

- **Roll-up of ancillary LLM cost** (contextual-prefix step, judge/RAGAS) — SP7 corrects answer-gen cost only. A future slice can wrap those calls with `estimate_cost` and sum into a per-query total. Hook: `Usage.__add__` already sums `cost_usd`, so aggregation is mechanically ready.
- **Authoritative pricing source** — `PRICING` remains a hand-maintained estimate. Open question: do we want a periodic reconciliation task (or a startup assertion) that fails loud if `gen_model`/`anthropic_model` isn't in `PRICING`? (Would turn defect #2 into a deploy-time error for *configured* models.)
- **Should `cost_priced=False` be a hard error in CI eval runs?** Proposed default is warn+flag; a stricter mode (fail the eval if any query is unpriced) is a cheap future toggle on top of the `cost_priced` flag.
- **Dated-key drift** — Anthropic may change id formats; the prefix rule (D4) is conservative but should be covered by a periodic check against a live `messages.create` response `model` string (an integration test gated on a key, out of this offline slice).
- **Langfuse detail keys, deployed-version check** — D5 pins `usage_details={"prompt_tokens","completion_tokens"}` and `cost_details={"input","output","total"}`, matching the SDK's first-party OpenAI integration (`langfuse/openai.py:1202`). The generic docstring at `client.py:260` shows the `{"total_cost": …}` alias. Both are accepted; before merge, eyeball one real trace in the deployed Langfuse UI to confirm the native Usage/Cost panels populate with the chosen breakdown keys.
