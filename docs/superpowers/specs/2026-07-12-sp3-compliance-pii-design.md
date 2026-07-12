# SP3 · Compliance / Data Protection — Design Spec

**Date:** 2026-07-12
**Status:** Approved (core decisions locked; optional salted value-hash added off-by-default per review) → adversarial spec-verification → writing-plans
**Program:** Production-hardening, Phase 0 (risk-ordered). Third slice, after SP1 (Security) and SP2 (Guardrails).

---

## 1. Context & problem

PII flows through the system unprotected (verified against source):

| # | Defect | Location | Reality |
|---|--------|----------|---------|
| 1 | PII never redacted at ingest | `ingest/run.py` (no `PIIRedactor` call) | Document PII flows cleartext into the contextual-prefix **LLM** (`:68`), the **embeddings** (`:79`), the **vector store** (`:87`), and the **BM25 pickle** (`:97`). Query-time redaction cannot undo data at rest. |
| 2 | Raw query traced pre-redaction | `core/pipeline.py:91` (root span `question=question`) | The raw question is recorded to Langfuse **before** input-guard redaction (`:112`), at 100% sampling, with no mask — so every query's PII lands in the observability backend. |
| 3 | Audit log is in-memory + stores raw PII | `ingest/pii.py:113` (`audit_log` list) + `:99` (`value`) | `PIIRedactor.audit_log` is lost on restart, **and** each finding stores the raw `value` — persisting it as-is would create a *new* PII store. |
| 4 | No output-side PII scan | (deferred from SP2) | Defense-in-depth backstop for PII in answers (echoed from the question, model-generated, or — in `keep` mode — retrieved from a document). |
| 5 | Detector is narrow US-regex | `ingest/pii.py:22-61` | Only email/phone/SSN/credit-card; cannot catch names, addresses, or locations. |

## 2. Goals

- Give the operator a first-class, controllable **PII policy** — redact by default, but able to **deliberately index with PII** when they choose.
- In the default (`redact`) policy, PII never reaches the LLM, embeddings, or any store.
- Detection is **pluggable** (regex default, NER seam) and always runs, so the system always *knows* where PII is — even when it keeps it.
- A **durable** audit trail that records what PII exists and where, **without ever storing the raw value**.
- Observability never persists raw PII (mask + trace-after-redaction + sampling).
- All defaults are the safe posture; keeping PII is an explicit opt-in.

## 3. Non-goals (deferred)

- **Reversible tokenization / a PII vault** — chosen against (destructive redaction). A future option only if answers must return real PII to permitted callers.
- **NER as the default detector** — Presidio is an optional seam, not the default (heavy dependency).
- **Query-time PII gating** (retrieve/mask PII chunks by caller permission) — a future hook seeded by `keep`-mode tags; needs richer policy.
- **Per-tenant PII policy** (org A redacts, org B keeps) — deferred to **SP1.5 (Org & corpus management)**, which supplies the tenant registry a policy would attach to. SP3 ships **global config + per-run CLI**.
- **Re-ingesting existing corpora** to scrub already-stored PII — an **operational note** (a fresh ingest is redacted), not code in this slice.
- **Cost/usage tracing correctness** → SP7. SP3 owns only the PII *masking* + sample-rate on the tracer.

---

## 4. Decisions locked

| Decision | Choice |
|---|---|
| PII action | **Destructive redaction at ingest** — replace with `[TYPE]` placeholders before contextual prefixing, embedding, and storage. |
| Policy control | **`pii_mode` = `redact` (default) \| `keep`**, global config **+ a `--pii-mode` CLI override** on `ingest.run`. |
| `keep` mode | Index **with** PII, but **detection still runs**: tag `chunk.metadata["pii_types"]` and write the audit record ("know your data"). |
| Detector | **Pluggable `PIIDetector` Protocol**; `RegexPIIDetector` default (zero-dep); `PresidioPIIDetector` optional behind a `pii-ner` extra; selected by `pii_detector`. |
| Audit log | **Durable append-only JSONL**, records `{tenant_id, doc_id, chunk_id, type, start, end, ts}` — **type + location only, never the raw value**. |
| Audit value hash | **Optional, off by default.** When `pii_audit_value_hash` is on, each record also carries `value_hash = sha256(salt + value)` (truncated) so the *same* value can be correlated across documents **without storing plaintext**. Requires a secret `pii_audit_hash_salt`; enabling it without a salt **fails closed at boot**. |
| Ingest error policy | **Fail closed** — a detector error means the chunk is **not stored** (raise); in `keep` mode, an *audit-write* failure also aborts (never keep PII without a record). |
| Observability | **Langfuse `mask` callback + trace-the-question-only-after-redaction + `langfuse_sample_rate`** (all three). |
| Output scan | **`PIIGuardrail` added to the output guards** (REDACT path), gated by `pii_scan_output` (default `True`) — turn off when deliberately serving PII. |

---

## 5. Architecture & components

Follows the codebase's Protocol + registry pattern; detection separated from redaction from audit so each unit is single-purpose and testable.

### 5.1 `PIISpan` type + `PIIDetector` Protocol
- `core/types.py`: `PIISpan(type: str, start: int, end: int)` — **span only, no raw value** (nothing built from spans can leak PII).
- `core/interfaces.py`:
```python
class PIIDetector(Protocol):
    def detect(self, text: str) -> list[PIISpan]: ...
```

### 5.2 Detectors — `providers/pii/`
- `regex_detector.py` · `RegexPIIDetector` — the existing patterns (email/phone/SSN/card), refactored to return `PIISpan`s. Default; no dependencies.
- `presidio_detector.py` · `PresidioPIIDetector` — optional (`pii-ner` extra: `presidio-analyzer`, `spacy`); wraps Presidio's `AnalyzerEngine` → `PIISpan`s (names, addresses, locations, dates). Import is lazy so the default install never needs it.
- `build_pii_detector(settings)` in `core/registry.py` — the only place the concrete class is named.

### 5.3 Redaction + audit — `ingest/pii.py`, `ingest/audit.py`
- `redact(text, spans) -> str` — applies `[TYPE]` placeholders right-to-left (offset-safe). Pure.
- `PIIRedactor` (facade, keeps the name so `guardrails/pii_guard.py` is unchanged) — detect (via the configured detector) → redact → returns `(clean, findings)`; used by the input/output guards, where the action is always "redact".
- `ingest/audit.py` · `PIIAuditLog` — an append-only JSONL sink at `pii_audit_log_path`; `record(tenant_id, doc_id, chunk_id, text, spans)` writes one line per span with **no raw value**. Durable, queryable, not itself a PII store. When `pii_audit_value_hash` is enabled, it slices `text[span.start:span.end]` to compute `value_hash = sha256(salt + value)[:16]`, writes the hash, and **discards the slice** — the raw value is never persisted. `PIISpan` stays value-free (§5.1); the source text is passed to `record` only so the hash can be computed transiently, and is unused when hashing is off.

### 5.4 Ingest policy step — `ingest/run.py`
After chunking, before contextual prefixing/embedding, a `_apply_pii_policy(chunks, mode, detector, audit)` step:
- **`redact` mode:** replace each `chunk.text` with the redacted text; audit each finding.
- **`keep` mode:** leave `chunk.text` intact; set `chunk.metadata["pii_types"] = sorted({span.type ...})`; audit each finding.
- Both modes: **detection always runs.** Fail-closed on detector error (raise, don't store); in `keep` mode, an audit-write failure also aborts.

### 5.5 Observability PII protection — `observability/langfuse_tracing.py` + `core/pipeline.py`
- The `Langfuse(...)` constructor gains a `mask=` callback (runs `PIIRedactor` over any traced string; fail-closed → placeholder on mask error) and `sample_rate=settings.langfuse_sample_rate`.
- `pipeline.answer()` stops passing the raw `question` to the root span before the input guard runs (trace the redacted form, or omit until after `apply_redactions`).

### 5.6 Output-side scan — `guardrails/runner.py`
`default_runner` appends `PIIGuardrail()` to `output_guards` when `pii_scan_output` — any PII in the answer is REDACTED before return via the existing REDACT path (unaffected by SP2's BLOCK suppression).

---

## 6. Data flow

```
INGEST (pii_mode):
  load → chunk → detect(chunk.text)
       ├ redact: chunk.text := redact(...)            → audit(type/loc, no value)
       └ keep:   chunk.metadata["pii_types"] := [...]  → audit(type/loc, no value)
       → contextual prefix (LLM) → embed → vector store + BM25   (redacted in `redact` mode)

QUERY:
  question → input guards (injection / PII REDACT) → retrieval → generation
           → output guards (+ PII REDACT if pii_scan_output) → response
  Langfuse: mask() redacts every traced string; question traced only after redaction; sample_rate applied
```

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `pii_mode` | `"redact"` | `redact` \| `keep` (index with PII, tagged) |
| `pii_detector` | `"regex"` | `regex` \| `presidio` |
| `pii_audit_log_path` | `".audit/pii_audit.jsonl"` | durable audit sink |
| `pii_audit_value_hash` | `False` | add a salted hash per finding for cross-doc correlation (never the raw value) |
| `pii_audit_hash_salt` | `None` | secret salt for the value hash; **required** when `pii_audit_value_hash` is on |
| `pii_scan_output` | `True` | redact PII in answers |
| `langfuse_sample_rate` | `1.0` | trace sampling for high volume |

`ingest.run` gains `--pii-mode redact|keep` (overrides the config for that run). New optional extra: `pii-ner` (`presidio-analyzer`, `spacy`).

## 8. Error handling

- **Ingest detector error → fail closed:** the chunk is not stored (raise) — never silently persist un-redacted PII.
- **`keep`-mode audit-write failure → fail closed:** abort — never keep PII without an audit record. (In `redact` mode the data is already safe, so an audit-write failure logs an error and continues.)
- **Langfuse mask error → fail closed:** emit a placeholder, never the raw value.
- **Value-hash enabled without a salt → fail closed at boot:** `pii_audit_value_hash=True` with an empty `pii_audit_hash_salt` is rejected at startup — an unsalted hash of enumerable PII (an SSN, a card number) is trivially reversed by brute force, so we refuse to write one. (Even salted, this is a documented, off-by-default tradeoff: correlation without plaintext, but a leaked salt weakens known-format values.)
- The `.audit/` directory is created on first write and is gitignored.

## 9. Testing (TDD)

Offline (fakes, no network):
- **Ingest redact:** a doc with email/SSN → the stored `chunk.text`, `embed_text`, the upserted vector-store payload, and the BM25 pickle contain `[EMAIL]`/`[SSN]` placeholders and **zero** raw PII.
- **Ingest keep:** same doc → `chunk.text` retains the PII, `chunk.metadata["pii_types"]` lists the types, and an audit record is written.
- **Detector Protocol:** `RegexPIIDetector.detect` returns `PIISpan`s (no value field); `redact` applies placeholders; `PresidioPIIDetector` test skipped when the extra isn't installed.
- **Audit log:** JSONL is durable (survives a new `PIIAuditLog` instance), keyed by ids, and contains **no raw value**.
- **Audit value hash:** with `pii_audit_value_hash=True` and a salt, two docs containing the same email produce **identical** `value_hash` values and a **different** hash from a third email; the raw value never appears in the JSONL; with hashing off, no `value_hash` field is written. Boot validation rejects hash-enabled-without-salt.
- **Langfuse mask:** the mask callback redacts PII in a traced string; the root span never receives the raw question.
- **Output scan:** an answer containing an email → `[EMAIL]` before return.
- **Fail-closed:** a detector that raises at ingest → the chunk is not stored (exception propagates); `keep`-mode audit failure aborts.

## 10. Files

**Create:** `providers/pii/__init__.py`, `providers/pii/regex_detector.py`, `providers/pii/presidio_detector.py`, `ingest/audit.py`, `tests/test_pii_compliance.py`.
**Modify:** `core/types.py` (`PIISpan`), `core/interfaces.py` (`PIIDetector`), `core/config.py` (knobs), `core/registry.py` (`build_pii_detector`), `ingest/pii.py` (refactor to detector + `redact(text, spans)`, keep `PIIRedactor` facade), `ingest/run.py` (policy step + `--pii-mode`), `observability/langfuse_tracing.py` (mask + sample_rate), `core/pipeline.py` (trace after redaction), `guardrails/runner.py` (output PII guard), `pyproject.toml` (`pii-ner` extra), `.gitignore` (`.audit/`).

## 11. Open questions / future hooks

- **Per-tenant PII policy** → SP1.5 (attach `pii_mode` to a tenant in the registry).
- **Query-time PII gating** → use `keep`-mode `pii_types` tags to filter/mask by caller permission.
- **Reversible tokenization / vault** → only if a use case needs authorized detokenization in answers.
- **Salted value-hash** → **now in scope, off by default** (`pii_audit_value_hash` + `pii_audit_hash_salt`); see §4/§5.3/§8. Future extension: per-tenant salt once SP1.5 supplies the tenant registry.
