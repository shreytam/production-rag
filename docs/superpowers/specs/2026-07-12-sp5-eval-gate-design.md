# SP5 · Eval Gate That Gates — Design Spec

**Date:** 2026-07-12
**Status:** DRAFT — pending user review (design decisions are PROPOSED, not yet approved)
**Program:** Production-hardening, Phase 1. Turns the CI eval gate from a permanently-red, ignored job into a real merge blocker, and forces the security-critical ACL isolation suite to actually execute in CI. Depends on SP1 (auth) for the token that authenticates baseline generation; hands the "run the real store filters in CI" requirement that SP1 §11 flagged. Excludes cost accounting (SP7) and resilience/retries (SP6).

---

## 1. Context & problem

Every defect below was verified by reading the current code at the cited path:line.

| # | Defect | Location (verified) | Why it defeats the gate |
|---|---|---|---|
| 1 | `eval/baselines/` is **empty** — no committed baseline artifact exists. | `eval/baselines/` (directory empty; only `hotpotqa.baseline.results.json` exists under `eval/runs/`, which is git-ignored/ephemeral) | The gate has nothing to compare against. |
| 2 | Missing baseline → hard `sys.exit(1)` before any comparison. | `eval/compare.py:82-85` | With no committed baseline the compare step **always** exits 1, the eval job is **always** red, so the team learns to ignore it and real regressions merge. |
| 3 | Base vs new are **non-paired / different-N**: the paired bootstrap only runs when `len(base_items) == len(new_items)`; otherwise CI bounds are silently set to NaN. Base is run at `--limit 50`, new at `--fast` (15 items) — different sets, so this branch is never taken in CI. | `eval/compare.py:106-109`; item-count divergence between the ingest/eval steps in `.github/workflows/eval-gate.yml:130-147` | The statistical machinery exists but is dead on the real CI path. |
| 4 | The gate is **point-delta only** — the bootstrap CI is computed and printed but never used to pass/fail. | `eval/compare.py:107` (CI computed) vs `eval/compare.py:113` (gate uses `new_val < base_val - tolerance`, a scalar comparison ignoring `lo`/`hi`) | Noise on 15 items swings a point mean past tolerance → flaky both ways; a real but noisy regression can pass. |
| 5 | A **NaN metric silently passes**. Missing metrics default to `float("nan")`; `nan < base - tol` is `False`. | `eval/compare.py:98-99`, `:113` | A metric that failed to compute (judge crash, empty parse) is treated as "not a regression" and merges. |
| 6 | `answer_relevancy` compares a **query-space** embedding to **passage-space** embeddings (asymmetric-model mismatch). | `eval/generation_metrics.py:199` (`embedder.embed_query(question)`) vs `:200` (`embedder.embed_documents(generated)`); the store confirms these are different spaces — `providers/embedders/openai_compatible.py:56-62` sends `input_type="passage"` for documents, `"query"` for queries on NIM E5 models | The metric's cosine similarity is systematically depressed/biased, so its baseline is meaningless and it can't gate. |
| 7 | The **LLM judge is non-deterministic**: `Generator.complete()` accepts **no `seed`**, only `temperature` (default 0.0), and never passes a seed to the API. | `core/interfaces.py:80-87`; `providers/generators/openai_compatible.py:38-53` (kwargs has `temperature`, no `seed`); `eval/llm_judge.py:62-77` (single call, no majority vote) | Baseline and new-run judge scores drift run-to-run → CI gate flaps independent of code changes. |
| 8 | Security-critical **ACL live tests self-skip when no DB is present**, and **CI never provides one**. The `lint` job has no service containers and runs `pytest` with Qdrant/Postgres down (→ all live ACL tests skip); the `eval` job runs Qdrant but **runs no pytest at all**. | `tests/test_stores_acl.py:334-340,424-430`; `tests/test_multitenant_isolation.py:135-136,144-145`; `.github/workflows/eval-gate.yml:26-47` (lint: no `services:`), `:52-159` (eval: has Qdrant service but no pytest step) | The real `qdrant_filter`/`pg_where` cross-tenant isolation is **never exercised in CI**. A regression that leaks another org's corpus ships green. |
| 9 | `fast_subset(seed=0)` is deterministic, but the **baseline and the new run must sample the identical 15 items** for the pairing in defect 3 to hold — nothing today enforces that base and new used the same subset/seed. | `eval/fast_subset.py:11-20`; `eval/run_eval.py:211-212` | Even after committing a baseline, a subset drift silently breaks the pairing. |

Net effect: the eval gate is decorative and the tenant-isolation guarantee is untested in CI. This slice makes both real.

---

## 2. Goals

- **Commit a paired baseline** artifact scored over the **exact same `fast_subset` (n=15, seed=0)** the CI new-run uses, via a reproducible `make` target.
- **Gate on the paired-bootstrap difference-CI**: fail when the difference-CI **upper bound** (`hi` of `new − base`) is below `−tolerance` (a statistically-real regression), not on a bare point delta.
- **Treat missing or NaN metrics as HARD failures** (fail closed), never silent passes.
- **Make the judge reproducible**: seed the generator and take a **majority vote** over an odd number of judge calls.
- **Fix the `answer_relevancy` embedding-space mismatch** (embed reverse-questions in query space).
- **Run the REAL Qdrant + pgvector ACL isolation tests in CI** against ephemeral service containers, and **fail the build if the isolation suite SKIPS** (no silent skip on missing DB in CI).
- **Make the eval job a required check**, with a defined, safe behavior for fork PRs that lack the secret.
- **Hook for Custom Evaluations:** Ensure the evaluation harness has a clear entry point/extensibility to support custom domain-specific evaluation metrics and checks in subsequent iterations.

## 3. Non-goals (deferred) — owner named

- **Cost/token accounting, per-run \$ budgets, spend dashboards** → **SP7** (Cost & Observability). This spec caps item count for wall-time but does not account cost.
- **Retry/backoff, circuit-breaking, timeout tuning of NIM calls under load** → **SP6** (Resilience). We rely only on the existing `max_retries` in `providers/generators/openai_compatible.py:26`.
- **Auth mechanism itself (JWT verifier, dev signer, token minting)** → **SP1** (Security & Tenancy). SP5 *consumes* a token to authenticate baseline generation; it does not build auth.
- **New retrieval/generation metric definitions or a RAGAS-library swap** → out of scope; SP5 only fixes the `answer_relevancy` space bug, gates existing metrics, and sets up hooks for custom metrics.
- **Golden-dataset expansion / relabeling** → out of scope; SP5 gates over the existing `data/eval/hotpotqa.json` fixture.
- **Physical per-tenant namespace isolation** → **VDB-Decision / SP11**. SP5 tests the current pooled-filter model.

---

## 4. Decisions (PROPOSED)

Lead option is the best-practice choice; all are PROPOSED for the user to confirm/override.

| # | Decision | Choice (PROPOSED) | Rationale |
|---|---|---|---|
| D1 | Baseline scope | Score baseline over the **same `--fast` subset (n=15, seed=0)** as the CI new-run; commit as `eval/baselines/hotpotqa.json`. | Makes base/new **paired and equal-N** so the paired bootstrap is valid (fixes defects 1,3,9). |
| D2 | Gate statistic | Fail when **difference-CI upper bound `hi(new−base) < −tolerance`**. | A regression must be *statistically distinguishable*, not point noise (fixes defect 4). Conservative: passes a noisy-but-not-clearly-worse run, fails a clearly-worse one. |
| D3 | Missing/NaN metric | **HARD failure** — any metric present in base but missing/NaN in new, or any NaN CI bound where pairing was expected, fails the gate. | Fail closed; a metric that didn't compute is a regression signal, not a pass (fixes defect 5). |
| D4 | Judge determinism | Add a **`seed`** param threaded into `Generator.complete()`; run the judge **`judge_votes` (default 3) times and take the median score**; seed each call deterministically (`base_seed + i`). | Median-of-odd is robust to a single outlier call; seeded calls maximize reproducibility on providers that honor `seed` (fixes defect 7). |
| D5 | `answer_relevancy` fix | Embed the reverse-questions with **`embed_query`** (query space), same as the original question. | Both sides must live in the same asymmetric space for cosine to mean anything (fixes defect 6). |
| D6 | Baseline generation | New `make baseline` target → `eval/run_eval.py --fast --write-baseline`, which writes directly to `eval/baselines/<dataset>.json`; committed by a human after review. | One reproducible command; the artifact is a reviewed, version-controlled input, not a CI-mutated file. |
| D7 | ACL tests in CI | A dedicated **`acl-isolation` CI job** with **both** Qdrant and Postgres/pgvector service containers, running only the live isolation tests with **`RAG_REQUIRE_LIVE_STORES=1`**, which turns "skip on unreachable" into "**FAIL** on unreachable/skip". | The security guarantee must be *executed*, and a skip in CI must be a red build (fixes defect 8). |
| D8 | Fork-PR / no-secret path | Eval + baseline-generation steps run only when the `NVIDIA_API_KEY` secret is present; on secret-less fork PRs the eval job is **`neutral`/skipped with an explicit marker**, and a **required "eval-gate-status" gate job** fails the merge unless eval ran OR a maintainer applied a documented `eval-skip-approved` label. The **`acl-isolation` job needs no secret** (fake embeddings, tiny DIM) and is **always required**. | Never fail-open: a PR can't merge by simply lacking the secret. Security tests never depend on the secret. |
| D9 | Judge/eval provider in CI | Keep NIM `meta/llama-3.3-70b-instruct` as judge; allow an override to a **cheaper model** via existing `JUDGE_MODEL` env for fork/cost-constrained runs, regenerating the baseline for that model. | Reuses the existing role-based generator (`core/registry.py:70`); no new provider path. |

---

## 5. Architecture & components

Follows the existing pattern: contracts in `core/interfaces.py`, config in `core/config.py`, concrete wiring in `core/registry.py`. No parallel framework. Small, single-purpose units.

### 5.1 Determinism on the generator contract — `core/interfaces.py`
Extend the existing `Generator.complete()` signature with an optional `seed`:
```
def complete(self, messages, *, response_model=None, temperature=0.0,
             max_tokens=1024, seed: int | None = None) -> LLMResponse: ...
```
- `providers/generators/openai_compatible.py`: pass `seed` into `kwargs` **only when not None** (some endpoints reject unknown params; keep it opt-in). Reuses the existing json_schema/BadRequest fallback.
- `providers/generators/anthropic.py`: Anthropic has no `seed` param → **ignore `seed`** (document that determinism there rests on `temperature=0` + median vote).
- `tests/_fakes.py` `RecordingGenerator`: accept and record `seed` so vote/seed behavior is unit-testable offline.

### 5.2 Seeded majority-vote judge — `eval/llm_judge.py`
`holistic_judge(..., votes: int = 1, base_seed: int = 0)`:
- Calls `generator.complete(..., temperature=0.0, seed=base_seed + i)` for `i in range(votes)`.
- Returns the **median `score`** over votes and the rationale of the median call. `votes=1` preserves current behavior; `run_eval` passes `settings.judge_votes`.

### 5.3 `answer_relevancy` space fix — `eval/generation_metrics.py`
Replace `embedder.embed_documents(generated)` with a query-space embed of each reverse-question (loop `embed_query`, or a new `embed_queries` batch helper if we add one — **PROPOSED**: loop `embed_query` to avoid touching the `Embedder` Protocol). Original question stays `embed_query`. Now both vectors are query-space.

### 5.4 Gate logic — `eval/compare.py` (rewritten `compare()`)
Single-purpose gate, fail-closed:
1. **Both** base and new must exist; base is the committed `eval/baselines/<dataset>.json`, new is the fresh run. (Same as today but base now always exists — D1/D6.)
2. Assert **paired equal-N** per metric. If a metric's `base_items`/`new_items` differ in length or are empty → **hard fail** (was a silent NaN, defect 3/5).
3. For each metric compute `paired_bootstrap(base_items, new_items)` → `(mean_diff, lo, hi)` from `eval/stats.py:44` (unchanged; already seeded).
4. **Gate rule (D2):** metric FAILS iff `hi < -tolerance` **or** `mean_diff` is NaN **or** either bound is NaN **or** the metric is absent from new. (Fail closed on any NaN — defect 5.)
5. Keep the existing fixed-width table (`eval/compare.py:49`) but add a `PASS/FAIL` column per metric and print the CI-based verdict.

### 5.5 Live-store CI harness — `tests/conftest.py` + existing tests
- New env-gated helper `require_live_or_fail(reachable: bool, backend: str)`:
  - If `RAG_REQUIRE_LIVE_STORES` is truthy and the backend is unreachable → **`pytest.fail(...)`** (not skip).
  - Else preserve today's `pytest.skip`.
- Swap the raw `pytest.skip` calls in `tests/test_stores_acl.py:340,430` and `tests/test_multitenant_isolation.py:136,145` to route through this helper. Offline dev behavior is unchanged; CI (with the flag set) turns an unreachable/skipped isolation suite into a **red build** (D7/defect 8).
- No new Protocol needed here — this is test infrastructure, not a swappable runtime component.

### 5.6 Baseline generator — `eval/run_eval.py`
Add `--write-baseline`: when set, write the run JSON to `eval/baselines/<dataset>.json` instead of `eval/runs/...`, and **refuse** unless `--fast` is also set (baseline must match the gated subset — D1/D9). Records `fast_subset` `n`/`seed`, `judge_votes`, and model versions in the artifact for auditability (extends the existing provenance block at `eval/run_eval.py:232-241`).

### 5.7 CI wiring — `.github/workflows/eval-gate.yml`
Three jobs (up from two):
- **`lint`** — unchanged (offline pytest with fakes).
- **`acl-isolation`** (new, **no secret**) — Qdrant **and** Postgres/pgvector services; `env: RAG_REQUIRE_LIVE_STORES=1`; runs `pytest tests/test_stores_acl.py tests/test_multitenant_isolation.py -q`. Skip/unreachable ⇒ fail.
- **`eval`** — gated on `secrets.NVIDIA_API_KEY` presence; ingest → run_eval `--fast` → `compare` against committed baseline. On missing secret: emit a `neutral` outcome.
- **`eval-gate-status`** (new, tiny, `needs: [eval]`, `if: always()`) — the **required** check: passes only if `eval` succeeded, or the PR carries the maintainer `eval-skip-approved` label. Never passes merely because the secret was absent (D8).

---

## 6. Data flow

**Baseline creation (offline, human-run once per intended change to the reference):**
```
make baseline DATASET=hotpotqa
  → ingest.run --dataset hotpotqa --limit 50           (needs NVIDIA_API_KEY; SP1 token if API-gated)
  → eval.run_eval --dataset hotpotqa --version full --fast --write-baseline
        fast_subset(n=15, seed=0)  →  15 items
        per item: pipeline.run → retrieval + generation metrics + seeded median-vote judge
  → writes eval/baselines/hotpotqa.json   (provenance: subset n/seed, judge_votes, git_sha, models)
  → human reviews + commits the JSON
```

**CI on every PR:**
```
job acl-isolation (no secret):
  qdrant + pgvector services up
  RAG_REQUIRE_LIVE_STORES=1 pytest <acl tests>
     reachable → real qdrant_filter / pg_where cross-tenant asserts RUN
     unreachable/skip → FAIL the job

job eval (only if NVIDIA_API_KEY present):
  ingest --limit 50 → run_eval --fast (SAME subset n=15 seed=0 as baseline)
  compare(base = committed baselines/hotpotqa.json, new = fresh run):
     per metric: paired_bootstrap(base_items, new_items) → (diff, lo, hi)
       equal-N assert (else HARD FAIL)
       FAIL iff hi < -tolerance OR any NaN OR metric missing
  → exit 0/1

job eval-gate-status (required):
  pass iff eval succeeded OR label 'eval-skip-approved'
```

---

## 7. Config knobs (`core/config.py`)

| Knob | Default | Purpose |
|---|---|---|
| `judge_votes` | `3` | Odd number of seeded judge calls; median score taken. `1` = legacy single call. |
| `judge_seed` | `0` | Base seed for judge vote calls (`seed = judge_seed + i`). |
| `eval_tolerance` | `0.03` | Regression tolerance; gate fails when difference-CI `hi < -eval_tolerance`. Matches the current workflow's `--tolerance 0.03`. |
| `eval_fast_n` | `15` | Size of the gated `fast_subset`; baseline and CI run must share this. |
| `eval_fast_seed` | `0` | Seed for `fast_subset`; baseline and CI run must share this. |
| `eval_bootstrap_resamples` | `1000` | Bootstrap resample count (already the `stats.py` default; surfaced for reproducibility). |
| `require_live_stores` | `False` | When true (CI `acl-isolation` sets `RAG_REQUIRE_LIVE_STORES=1`), unreachable/skipped live ACL tests FAIL instead of skip. |

CI-only env (not new `Settings` fields): `RAG_REQUIRE_LIVE_STORES` (maps to `require_live_stores`), and the eval job continues to set the NIM `EMBED_*/GEN_*/JUDGE_*` envs already present in the workflow.

---

## 8. Error handling — fail closed on security & correctness paths

- **Missing committed baseline** → still `exit 1`, but D6 guarantees the file exists; the message now points to `make baseline` rather than a dead-end.
- **Unequal-N / empty per-metric item lists** where pairing is required → **hard fail** with the offending metric named (was silent NaN).
- **NaN metric or NaN CI bound** → **hard fail** (defect 5). Never `nan < x` short-circuiting to pass.
- **Metric present in base, absent in new** → **hard fail** (a metric that stopped being produced is a regression).
- **Live ACL store unreachable in CI** (`require_live_stores`) → `pytest.fail`, red build (defect 8). Locally (flag unset) → skip, unchanged.
- **Judge parse failure** in a vote → that vote is dropped; if **all** votes fail, the item's judge score is treated as a **missing metric** → hard fail at the gate (never a silent 0.0 that could mask regression).
- **Fork PR without secret** → eval `neutral`, and the required `eval-gate-status` job **fails** unless explicitly labeled by a maintainer (never fail-open on merge).
- **Seed not honored by provider** → determinism degrades gracefully to median-vote-at-temp-0; documented, not fatal.

---

## 9. Testing (TDD) — offline, deterministic

Red-first, all runnable without a network or real DB (except the CI-only live job):

**Gate logic (`tests/test_eval_compare.py`, new):**
- Synthetic base/new JSONs with paired equal-N items. A metric whose difference-CI `hi < -tolerance` → gate FAILS; a noisy metric whose `hi >= -tolerance` → PASSES.
- Unequal-N per metric → **hard fail** (assert non-zero, assert message names the metric).
- NaN metric in new → **hard fail** (regression-guards the exact `float("nan") < x` bug at `eval/compare.py:98-99,113`).
- Metric present in base, absent in new → hard fail.
- All-within-CI run → exit 0.

**Judge determinism (`tests/test_eval_metrics.py`, extend):**
- `RecordingGenerator` records seeds; assert `holistic_judge(votes=3, base_seed=0)` issues seeds `{0,1,2}` and returns the **median** of three stubbed scores.
- `votes=1` reproduces the current single-call behavior.

**`answer_relevancy` space (`tests/test_eval_metrics.py`, extend):**
- A `FakeEmbedder` that returns a **different vector for `embed_documents` vs `embed_query`** on the same text; assert the metric now calls `embed_query` for the generated questions (both sides query-space) — locks the fix at `eval/generation_metrics.py:200`.

**Generator seed plumbing (`tests/test_generators.py` or `_fakes`):**
- `complete(seed=7)` forwards `seed` into request kwargs for the OpenAI-compatible generator; Anthropic ignores it without error.

**Baseline writer (`tests/test_run_eval.py`, extend):**
- `--write-baseline` requires `--fast` (error otherwise); writes to `eval/baselines/<dataset>.json`; artifact records `fast_n`, `fast_seed`, `judge_votes`.

**Live-ACL fail-closed harness (`tests/test_live_gate.py`, new, offline):**
- With `require_live_stores=True` and a stubbed "unreachable" backend, `require_live_or_fail` **fails** (not skips); with flag off, it **skips**. (The real Qdrant/pgvector round-trip runs only in the CI `acl-isolation` job.)

---

## 10. Files

**Create:**
- `eval/baselines/hotpotqa.json` — committed paired baseline (generated by `make baseline`, human-reviewed).
- `tests/test_eval_compare.py` — gate-logic tests.
- `tests/test_live_gate.py` — fail-closed live-store harness test.
- (optional) `tests/test_run_eval.py` — if not already present, baseline-writer tests.

**Modify:**
- `eval/compare.py` — CI-based fail-closed gate (§5.4, D2/D3), PASS/FAIL table column.
- `eval/generation_metrics.py:200` — `answer_relevancy` query-space embed (D5).
- `eval/llm_judge.py` — seeded median-vote judge (D4).
- `eval/run_eval.py` — `--write-baseline`, pass `judge_votes`, thread `eval_fast_n/seed`, extend provenance.
- `eval/fast_subset.py` — accept `n`/`seed` from config (already parameterized; wire defaults from `core/config.py`).
- `core/interfaces.py` — add `seed` to `Generator.complete`.
- `providers/generators/openai_compatible.py` — forward `seed` when not None.
- `providers/generators/anthropic.py` — accept + ignore `seed`.
- `tests/_fakes.py` — `RecordingGenerator` records `seed`; `FakeEmbedder` distinguishes query vs document space for the metric test.
- `core/config.py` — knobs in §7.
- `core/registry.py` — no new component, but pass `judge_votes`/`judge_seed` through where the judge generator is built for eval (`build_generator(role="judge")`, `core/registry.py:70`).
- `tests/test_stores_acl.py`, `tests/test_multitenant_isolation.py` — route live skips through `require_live_or_fail`.
- `tests/conftest.py` — `require_live_or_fail` helper + `require_live_stores` plumbing.
- `.github/workflows/eval-gate.yml` — add `acl-isolation` job (Qdrant + pgvector, no secret), gate the `eval` job on secret presence, add required `eval-gate-status` job (§5.7).
- `Makefile` — `baseline` target (`.PHONY`), reusing the existing `DATASET` var.
- `docs/architecture.md` — replace the stale "Bootstrapping the baseline" note referenced at `.github/workflows/eval-gate.yml:13-14` with the `make baseline` procedure.

---

## 11. Open questions / future hooks

- **Q1 — tolerance per metric?** A single `eval_tolerance` (0.03) treats faithfulness and recall@5 identically. A per-metric tolerance map may be warranted once we have baseline variance data. **PROPOSED:** single tolerance now; revisit after first 10 CI runs.
- **Q2 — n=15 statistical power.** A difference-CI on 15 paired items is wide; it will only catch sizable regressions. Enlarging the gated subset trades CI cost (SP7) for sensitivity. **PROPOSED:** keep 15; document the sensitivity floor.
- **Q3 — judge model drift.** The baseline is model-specific; a NIM model version bump silently invalidates it. Hook: record `judge_model` in the artifact (done) and add a CI assertion that the run's `judge_model` matches the baseline's, failing on mismatch. **PROPOSED as a fast-follow.**
- **Q4 — provider `seed` support.** If NIM ignores `seed`, D4 reduces to median-vote-at-temp-0. Confirm empirically during baseline generation; if variance is still high, raise `judge_votes` to 5.
- **Q5 — flaky live services in CI.** The `acl-isolation` job must not become a new "always red" job (the very failure mode we're fixing). Hook: rely on the service `--health-*` options already modeled in `.github/workflows/eval-gate.yml:64-68` and a bounded connect timeout; a genuinely-down service is a real failure worth surfacing.
- **Q6 — baseline provenance vs. `--limit 50` ingest.** The corpus is ingested at `--limit 50` but the gate scores a 15-item `fast_subset`; ensure the subset's relevant chunks are within the ingested 50. Hook: `run_eval` could warn if any golden `relevant_chunk_ids` are absent from the ingested corpus.
- **Q7 — custom evaluations integration.** We must facilitate adding domain-specific custom checks (e.g., toxicity, safety, groundness) that can be run concurrently with standard RAGAS metrics. Hook: add a registration mechanism for custom metric functions in the evaluation runner that are called and evaluated under the same bootstrap comparison logic.
