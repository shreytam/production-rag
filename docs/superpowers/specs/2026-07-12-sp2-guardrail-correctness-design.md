# SP2 · Guardrail Correctness — Design Spec

**Date:** 2026-07-12
**Status:** Approved (design), pending spec review → writing-plans
**Program:** Production-hardening, Phase 0 (risk-ordered). Second slice, after SP1 (Security & Tenancy).
**Revision:** rev-2 — hardened after adversarial spec verification (fixed: injection normalization that couldn't match spaced attacks; a cosmetic groundedness timeout; a phantom-citation false-positive regression on legit brackets; an object-level content leak via `metadata`).

---

## 1. Context & problem

The guardrails are wired into the live pipeline but several are inert or bypassable at the exact point they matter (verified against source). SP2 makes them actually enforce. The six confirmed defects:

| # | Defect | Location | Reality |
|---|--------|----------|---------|
| 1 | Blocked output returned verbatim | `core/pipeline.py:152-158` | `apply_redactions` acts only on REDACT results; output guards emit BLOCK, so `ans.text`/`citations`/`contexts` are never suppressed. Sets `refused=True` but returns the offending content. |
| 2 | Phantom citations bypass the guard | `generation/grounded_generator.py:62-71` + `guardrails/citation_enforcement.py:54` | Generator silently drops markers absent from `marker_map`; a claimed `[99]` disappears from `citations` but the model's claim is never rejected. CitationGuardrail only checks resolved `answer.citations`, never the model's raw claimed markers. |
| 3 | Injection LLM second-opinion is dead code | `guardrails/input_injection.py:51,58-79` | `generator` stored but never used; `check()` is regex-only with no input normalization, so `i g n o r e` / unicode / leetspeak bypass every pattern. |
| 4 | Indirect injection never scanned | pipeline path | The injection guard runs on user input only; a poisoned retrieved chunk is never inspected. |
| 5 | Groundedness passes on empty context | `guardrails/output_groundedness.py:47-55` | Empty `contexts` → PASS, so a non-refused answer with no supporting context (a hallucination) is not blocked. |
| 6 | Runner has no error handling / no fail policy | `guardrails/runner.py:100-106` | `guard.check` is called with no try/except; any guard exception (esp. groundedness's in-band LLM calls) 500s the request. No documented fail-open/closed stance. |

## 2. Goals

- A blocked answer surfaces **only** a generic refusal — never the offending text, citations, or contexts, **and no residual copy of them on the returned object** (`metadata`).
- The model cannot present a fabricated citation (a claimed marker with no real passage) to the caller.
- Guardrail exceptions follow an explicit **per-guard fail policy** and can never 500 a request; the groundedness LLM call is bounded by a real wall-clock timeout.
- Input injection detection resists trivial obfuscation (letter-spacing, unicode, leetspeak) and uses an LLM second-opinion **only** on ambiguous input.
- Poisoned retrieved documents are detected + logged (defense-in-depth), without a self-inflicted DoS.
- A non-refused answer with no supporting context is blocked.
- **No new false-positive regressions** — legitimate bracketed text (`[2020]`, `arr[0]`) and benign role phrasing (`act as a translator`) must not be blocked.

## 3. Non-goals (deferred)

- **Output-side PII scanning** and Langfuse raw-query redaction → SP3 (Compliance). SP2 keeps the existing input PII REDACT behavior only.
- **Global exception handler / request-id** → SP9. SP2's runner handles *guardrail* exceptions only.
- **New guard types** beyond the indirect-injection scan function. No NER-based PII, no new output guards.
- **Auth** (SP1) — assumed present; SP2 does not touch identity.
- **Cancelling an in-flight LLM call** — not possible for a running future; the groundedness timeout abandons the call (see §5.4), it does not cancel it.

---

## 4. Decisions locked

| Decision | Choice |
|---|---|
| Fail policy on guard **exception** | **Per-guard**: deterministic safety guards (injection, PII, citation, schema) fail **closed** (→ BLOCK); groundedness fails **soft** (→ PASS + `groundedness_unverified` flag). Every exception recorded in `result.metadata["error"]`. The runner catches **exceptions**, not timeouts — each guard that makes an LLM call owns its own timeout. |
| Indirect injection (retrieved docs) | **Detect + spotlight + backstop + log** — flag & log on a hit, no hard block. Always-on (no config knob) by design. Spotlighting already exists in `prompts.py`; output guards are the backstop. |
| Input injection guard | **Normalize (two forms) + tiered + escalate on borderline** — match space-flexible patterns against both a whitespace-collapsed and a whitespace-stripped form; strong hit → BLOCK (no LLM); no hit → PASS (no LLM); weak/ambiguous-only → one LLM classifier call when enabled+available, else BLOCK (fail closed). |
| Output-block response | Content **and its metadata copies** suppressed; caller gets a fixed generic refusal. The block *reason* is kept only in the trace/log, and is **stripped from the returned object** (attacker-hardening). |
| Groundedness LLM budget | Real wall-clock bound via a **module-level executor** + `groundedness_timeout_seconds` (default **20.0s**); on timeout the call is **abandoned** (runs to completion in the background — documented thread/cost tradeoff) and the guard soft-fails. |
| Phantom-citation scope | Verify the model's **claimed citation markers** (from structured output) against the valid passage set — NOT arbitrary bracketed text — so legit `[2020]`/`arr[0]` never trigger a block. |

---

## 5. Architecture & components

No change to the `Guardrail` Protocol shape or the runner's public API. Changes are behavioral, plus one optional guard attribute:

- **`fail_closed: bool`** — an optional attribute the runner reads via `getattr(guard, "fail_closed", True)`. Deterministic guards inherit the default `True`; `GroundednessGuardrail` sets `False`.
- **`scan_for_injection(text) -> list[str]`** — a module-level helper in `guardrails/input_injection.py` (normalized pattern matching) shared by the input guard and the pipeline's retrieved-doc scan.

### 5.1 Runner error handling + per-guard fail policy — `guardrails/runner.py`
`_timed_check` wraps the call in try/except (exceptions only — no runner-level timeout; deterministic guards are fast, and the two guards that call an LLM each bound their own call):

```python
@staticmethod
def _timed_check(guard, text, context=None):
    t0 = time.perf_counter()
    try:
        result = guard.check(text, context=context)
    except Exception as e:  # a guard must never 500 the request
        fail_closed = getattr(guard, "fail_closed", True)
        result = GuardrailResult(
            name=getattr(guard, "name", "unknown"),
            action=GuardrailAction.BLOCK if fail_closed else GuardrailAction.PASS,
            reason=f"guard errored: {type(e).__name__}",
            metadata={"error": str(e), "groundedness_unverified": not fail_closed},
        )
    result.metadata["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return result
```

### 5.2 Output-block content + metadata suppression — `core/pipeline.py`
The block branch (lines 153-158) still sets `refused=True` + `blocked_by`; the *reason* is written to the trace/log only. Suppression runs at the **tail** of `answer()`, after the retrieval-metadata stash (after line 176 today), keyed on the flag so nothing re-populates it. It scrubs the content **and every metadata copy of it** (this closes the object-level leak — `grounded_generator.py:83` stashes the full answer text in `metadata["structured_output"]`, and the guard log carries reason strings):

```python
if ans.metadata.get("blocked_by") == "output_guardrail":
    ans.text = OUTPUT_BLOCK_MESSAGE
    ans.citations = []
    ans.contexts = []
    ans.metadata["retrieved_doc_ids"] = []
    ans.metadata["retrieved_chunk_ids"] = []
    ans.metadata.pop("structured_output", None)     # held the full original text
    ans.metadata.pop("block_reason", None)          # reason stays in trace/log only
    # keep only action labels on the returned guard log — drop reason text
    gl = ans.metadata.get("guardrails", {})
    for phase_results in gl.values():
        for r in phase_results:
            r.pop("reason", None); r.pop("payload", None); r.pop("metadata", None)
return ans
```

**Client-facing artifact (defined):** the `run()` return dict — including `answer_obj` (the full `Answer`) — is the client contract this guarantee is asserted against. After suppression, neither `run()["answer"]` nor `answer_obj.text`/`.citations`/`.contexts`/`.metadata` contains the offending content or the block reason. (The HTTP `QueryResponse` in `app/api.py` already omits `metadata` — defense in depth, but the guarantee holds at the object level regardless.) `OUTPUT_BLOCK_MESSAGE` is a module constant, e.g. *"I can't provide an answer that passes the system's safety and grounding checks for this request."*

### 5.3 Phantom-citation catch — `grounded_generator.py` + `citation_enforcement.py`
Verify the model's **claimed** citation markers, not arbitrary text (avoids blocking legit `[2020]`/`arr[0]`/footnotes):

- The generator stashes, on the returned `Answer`:
  - `answer.metadata["valid_markers"] = sorted(marker_map.keys())` — the passage numbers that actually exist (1..N).
  - `answer.metadata["claimed_markers"] = sorted(set(parsed.citations))` when structured output parsed; on the **fallback path** (`resp.parsed is None`, model ignored the schema) set `claimed_markers = []` (we cannot distinguish a claimed citation from incidental prose, so there is nothing to verify — documented residual; the existing chunk-id and ≥1-citation checks still apply).
- `CitationGuardrail.check` (no text regex, no cross-module import):
  - If `valid_markers` **or** `claimed_markers` is absent from `answer.metadata` → **skip** the phantom check (treat as "cannot verify", so directly-constructed `Answer`s in tests don't spuriously block); retain the existing ≥1-citation and cited-chunk-id-in-context checks.
  - Else BLOCK when any marker in `claimed_markers` is not in `valid_markers` (catches the model claiming `[99]`, including mixed `[1]`+`[99]`).

### 5.4 Groundedness fixes — `output_groundedness.py`
- `fail_closed = False`.
- Empty-context branch: if `answer` is present and **not** `refused` → **BLOCK** ("non-refused answer has no supporting context"); if refused → PASS.
- **Real timeout via a module-level executor** (a `with`-block executor would `join` on exit and defeat the timeout — the call must be *abandoned*, not awaited):

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

# module-level, bounded so abandoned calls can't grow threads without limit
_GROUNDEDNESS_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="groundedness")

# inside check():
fut = _GROUNDEDNESS_POOL.submit(
    faithfulness, question=question, answer=text, contexts=contexts, generator=self._generator
)
try:
    score = fut.result(timeout=self._timeout_seconds)  # does NOT cancel on timeout
except FTimeout:
    return GuardrailResult(name=self.name, action=GuardrailAction.PASS,
        reason="groundedness check timed out",
        metadata={"groundedness_unverified": True})
```

**Documented tradeoff:** a Python future cannot be cancelled, so on timeout the `faithfulness` call keeps running in the background until its own LLM `request_timeout_seconds` elapses (a bounded thread + double API cost for that one request). This is the only way to actually bound the request's wall clock. The pool's `max_workers=4` caps concurrent abandoned work; operators may additionally lower the generator's request timeout for the groundedness role.

### 5.5 Input injection: normalize + tiered + escalate — `input_injection.py`
- **`_normalize(text)`** produces two forms so both normal and letter-spaced attacks match:
  - `base = casefold(NFKC(text))`, then strip zero-width chars (`​‌‍⁠﻿`), then apply a small leetspeak map (`0→o,1→i,3→e,4→a,5→s,7→t,@→a,$→s`).
  - **Form A (spaced):** `base` with runs of whitespace collapsed to a single space (word boundaries preserved).
  - **Form B (compact):** `base` with **all** whitespace removed (`ignorepreviousinstructions`) — this is what catches `i g n o r e   p r e v i o u s`.
- **Space-flexible patterns:** the inter-token `\s+` in every multiword pattern becomes `\s*` (zero-or-more), so a pattern matches whether the words are spaced (Form A) or fused (Form B). A pattern matching in **either** form counts as a hit. `\b`-anchored single-word patterns (e.g. `jailbreak`) rely on Form A.
- **Tiers.** Existing labels split into:
  - **strong** (unambiguous): `ignore_previous, disregard_above, forget_instructions, reveal_prompt, exfiltrate_instructions, jailbreak, dan_mode, override_rules, new_instructions, end_of_system, pretend_no_rules`.
  - **weak** (ambiguous): `system_prompt, you_are_now`. **`role_override` is narrowed** from the over-broad `act as a/an <anything>` to genuinely adversarial personas (`act as DAN`, `act as an unfiltered/uncensored AI`, `act as a hacker`, `with no restrictions`, `pretend to be the system`); generic `act as a translator/editor/reviewer` no longer matches at all.
- **Decision:** any **strong** match → BLOCK (no LLM). **No** match → PASS (no LLM). **Only weak** matches → borderline: if `injection_llm_escalation` and a generator is available → one LLM classifier call decides (BLOCK/PASS), bounded by the generator's request timeout; otherwise → BLOCK (fail closed). **Documented:** in an offline/no-generator deployment weak-only inputs fail closed (block); in the normal prod build a generator is always wired, so weak-only inputs are adjudicated by the LLM rather than auto-blocked.
- `scan_for_injection(text)` returns matched **strong** labels on the normalized forms (used by 5.6; only strong labels flag indirect injection to avoid noisy retrieval-time false positives).

### 5.6 Indirect-injection detection — `core/pipeline.py`
Concrete threading (the scan happens in the retrieval span; the flag lands on the `Answer` after generation). Always-on, no config knob (defense-in-depth default):

```python
# in the retrieval span, after `scored` is available:
suspected = sorted({lbl for sc in scored for lbl in scan_for_injection(sc.chunk.text)})
if suspected:
    s_ret.update(output={"n_hits": len(scored), "indirect_injection_suspected": suspected})
    logger.warning("indirect_injection_suspected: %s", suspected)
# ... after `ans` is produced by grounded.generate():
if suspected:
    ans.metadata["indirect_injection_suspected"] = suspected
```
No block — spotlighting (already in `prompts.py`) + the output guards are the backstop, per the locked decision.

---

## 6. Data flow

```
input → Injection(normalize×2 → space-flexible tiered regex → maybe 1 LLM) / PII(REDACT)
        └ strong (or weak-noLLM) block ⇒ refusal (generation never runs)
retrieval → scan_for_injection(chunks) → flag + log + trace   (NO block)
generation (spotlighted prompt, unchanged)
output → runner(try/except EXCEPTIONS, per-guard fail policy):
         Citation(claimed-marker check) · Schema · Groundedness(empty-ctx BLOCK, real timeout→soft-fail)
      → any BLOCK ⇒ suppress text/citations/contexts/ids + metadata copies, return OUTPUT_BLOCK_MESSAGE
```

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `injection_llm_escalation` | `True` | Enable the borderline LLM second-opinion (needs a generator) |
| `groundedness_timeout_seconds` | `20.0` | Real wall-clock bound on the groundedness call; timeout → soft-fail |

`OUTPUT_BLOCK_MESSAGE` and the groundedness executor `max_workers` are module constants, not knobs. The indirect-injection scan is intentionally always-on (no knob).

## 8. Error handling / fail policy

Per §4: the runner catches guard **exceptions** (not timeouts) — deterministic guards fail closed (BLOCK), groundedness fails soft (PASS + `groundedness_unverified`), all exceptions captured in `result.metadata["error"]`. **Timeouts are each guard's own responsibility:** groundedness bounds its `faithfulness` call via the module-level executor (§5.4); the injection LLM escalation call is bounded by the generator's `request_timeout_seconds`. No guardrail path can 500 or unboundedly hang a request.

## 9. Testing (TDD — red first)

Unit + integration, expanding `tests/test_guardrails.py` and `tests/test_prompt_injection.py`:

- **Output block suppression:** an answer that trips CitationGuardrail/GroundednessGuardrail comes back with `refused=True`, `answer == OUTPUT_BLOCK_MESSAGE`, empty `citations`/`contexts`/`retrieved_*_ids`, **and** `answer_obj.metadata` has no `structured_output`, no `block_reason`, and no reason strings in the guard log.
- **Phantom citation:** a parsed answer claiming `[99]` (not in `valid_markers`) → BLOCK; claiming only valid markers → PASS; mixed `[1]`+`[99]` → BLOCK. **No false blocks:** answer text containing `[2020]`, `arr[0]`, or a footnote `[12]` with 8 passages (none of them *claimed* citations) → PASS. Missing `valid_markers`/`claimed_markers` in metadata → phantom check skipped (existing checks still apply).
- **Runner fail policy:** a throwing deterministic guard → synthesized BLOCK; a throwing groundedness guard → PASS + `groundedness_unverified`; `metadata["error"]` set in both. A raised exception never propagates out of the runner (no-500 guarantee).
- **Groundedness:** non-refused answer with empty contexts → BLOCK; refused answer with empty contexts → PASS; a `faithfulness` stubbed to sleep beyond the timeout returns a soft-fail **within ~`groundedness_timeout_seconds`** (proves the wall-clock bound is real, not cosmetic); a genuinely unused in-range citation (answer cites `[3]` but contradicts passage 3) → BLOCK (backstop verified).
- **Injection normalization/tiers:** `ignore previous instructions`, `i g n o r e   p r e v i o u s   i n s t r u c t i o n s`, and a unicode/leetspeak variant all → BLOCK; a benign question and `act as a translator` → PASS with **zero** LLM calls; a weak-only input with escalation ON calls the generator exactly once; weak-only with escalation OFF / no generator → BLOCK.
- **Indirect injection:** a poisoned retrieved chunk (strong pattern) sets `indirect_injection_suspected` and is logged/traced but does **not** block the response.

## 10. Files

**Modify:** `guardrails/runner.py`, `guardrails/input_injection.py`, `guardrails/output_groundedness.py`, `guardrails/citation_enforcement.py`, `generation/grounded_generator.py`, `core/pipeline.py`, `core/config.py`, `tests/test_guardrails.py`, `tests/test_prompt_injection.py`. **Create:** none (all changes live in existing modules).

## 11. Open questions / future hooks

- **Output-side PII scanning** + Langfuse raw-query masking → SP3.
- **NER-backed PII / a dedicated injection-classifier model** (instead of the main generator) → future hardening.
- **Cancellable groundedness** — would require an async generator client or a cancellation token; today the timeout abandons the call. Revisit if abandoned-call cost becomes material.
- **Per-tenant guardrail policy** (different thresholds per org) → possible once the tenant registry (SP1.5) exists.
