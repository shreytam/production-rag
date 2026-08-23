# Audit: Evaluation Framework (`eval/`, corpus adapters, grounded generation)

**Scope:** `eval/*` (gate, experiment, metrics, judge, stats, backends, CLI), corpus adapters (`corpora/{arxiv,financebench,hotpotqa}/adapter.py`, `ingest/base.py`), and the eval-facing surface of `generation/grounded_generator.py` + `generation/prompts.py`. Cross-checked against CI wiring (`.github/workflows/eval-gate.yml`), config (`core/config.py`), tests (`tests/test_eval_*.py`, `tests/eval/fake_backend.py`), docs (`docs/architecture.md`), and historical design notes (`docs/superpowers/specs/2026-07-12-sp5-eval-gate-design.md`).

**Method:** Read-only source review. Every finding cites file:line evidence. Severity: 🔴 Critical / 🟠 High / 🟡 Medium / ⚪ Low. No code was modified, installed, or executed against live services.

**Status:** Complete (2026-08-24). 9 strengths, 11 defects (1 critical, 2 high, 5 medium, 3 low), 8 risks, prioritized recommendations.

---

## 1. System Overview

The evaluation framework is Langfuse-native: datasets, experiment runs, per-item traces,
and scores live on a hosted Langfuse instance; the repo supplies the task, the evaluators,
and the regression gate.

```
ingest.run --dataset hotpotqa --limit 50        # build corpus + data/eval/hotpotqa.json goldens
eval.dataset_cli seed                            # upload golden items to Langfuse
eval.experiment --fast                           # run pipeline on fast_subset(15), push traces+scores
eval.gate --new-run <r> --baseline-run baseline  # paired bootstrap / threshold gate; exit code
```

- **Backends:** `EvalBackend` Protocol (`eval/langfuse_eval.py:37-53`) with lazy `langfuse`
  import isolated in `eval/_langfuse_backend.py` (the only module importing it); offline
  tests use `tests/eval/fake_backend.py`.
- **Metrics:** pure retrieval metrics (`retrieval_metrics.py`: P@k, R@k, MRR, nDCG@k);
  RAGAS-style generation metrics with injected LLM/embedder (`generation_metrics.py`:
  faithfulness, answer_relevancy, context_precision, context_recall); holistic LLM judge
  with median-of-N voting (`llm_judge.py`). Optional real-RAGAS cross-check adapter
  (`ragas_adapter.py`) not imported by the core path.
- **Gate:** item-aligned paired-bootstrap (one-sided non-inferiority) and/or absolute
  floors (`gate.py`), configured via `core/config.py:150-161` (`eval_gate_mode=bootstrap`,
  `eval_tolerance=0.03`, `judge_votes=3`, `eval_fast_n=15`).
- **CI:** `.github/workflows/eval-gate.yml` — ingest → `--fast` experiment → gate vs the
  Langfuse run named `baseline`; an aggregation job turns skip/fail into the required check.

---

## 2. Strengths

**S1. Statistically sound regression gate — the old file-based gate's defects were fixed.**
The gate uses a *paired* bootstrap over per-item differences (resampling paired indices
preserves correlation between versions, `stats.py:63-72`) and fails when the upper CI of
delta < −tolerance (`gate.py:76-80`). A NaN CI is treated as **FAIL**, not pass
(`gate.py:78`) — directly closing the old "NaN silently passes" defect documented in
`docs/superpowers/specs/2026-07-12-sp5-eval-gate-design.md` (defects 3–5). Item sets and
metric sets must match exactly across runs or the gate raises instead of comparing
apples-to-oranges (`gate.py:25-28, 39-40`) — closing old defect 9.

**S2. Vacuous-pass guard on threshold mode.** Threshold/both modes require
`eval_gate_thresholds` to define at least one floor matching the run's metrics, otherwise
they raise (`gate.py:56-60`); tests pin this (`tests/test_eval_gate.py:76-89`). Commit
`1a19034` ("guard against vacuous threshold pass") shows this was deliberately hardened.

**S3. Clean dependency seams; fully offline-testable core.**
`EvalBackend` is a `runtime_checkable` Protocol; Langfuse is imported lazily only inside
the real backend (`langfuse_eval.py:4-7`, `_langfuse_backend.py:13`); every generation
metric takes injected `Generator`/`Embedder`; retrieval metrics are pure functions over
opaque ids. Result: ~10 gate tests, comprehensive metric tests, subset determinism tests,
backend round-trip tests — all network-free (`tests/test_eval_metrics.py`,
`test_eval_gate.py`, `test_eval_backend.py`, `test_eval_experiment.py`).

**S4. Metric suite is comprehensive and tracks RAGAS definitions.**
Faithfulness (atomic claims → verdict fraction), context_recall (ground-truth statements →
attributable fraction), context_precision (AP-weighted, `generation_metrics.py:247-254`),
answer_relevancy via reverse-question embedding similarity — including a documented fix to
embed both sides in query space for asymmetric models (`generation_metrics.py:199-200`).
Plus recall/precision/MRR/nDCG retrieval metrics and a groundedness/completeness/relevance
holistic judge. The optional `ragas_adapter.py` cross-checks native numbers against the
reference library (with NaN-safe aggregation, `ragas_adapter.py:154-157`).

**S5. Deterministic, seeded statistics.** Both bootstrap functions seed their own
`np.random.default_rng(seed)` per call so results are reproducible regardless of global RNG
state (`stats.py:36, 63`); percentile CIs; `fast_subset` is a seeded `random.sample`
(`fast_subset.py:19-20`). Judge votes thread deterministic seeds (`base_seed + i`,
`llm_judge.py:86`; verified by `test_sp5_llm_judge_vote.py:23`).

**S6. Eval configuration hygiene is explicit and commented.** Cache OFF (a cache hit would
silently confound metrics), guardrails OFF, rewriter ON because "eval must measure
query-rewriting's recall impact" (`experiment.py:82-85`, mirrored in
`ragas_adapter.py:130-138`). Judge runs on a separate role/model knob
(`build_generator(role="judge")`, `core/registry.py:85-96`).

**S7. Median-of-N judge voting.** `judge_votes=3` default with median aggregation is robust
to a single outlier vote (`llm_judge.py:94-97`, `config.py:160`).

**S8. Dataset curation loop includes production trace promotion.**
`dataset_cli add-from-trace` creates items with native `source_trace_id` provenance for
later human labeling (`_langfuse_backend.py:60-69`) — a best-practice flywheel from prod
traffic to golden set.

**S9. Gate output is a readable per-metric table with base/new/delta/CI/floor/verdict**
(`gate.py:62-91`), exiting nonzero on failure — good CI ergonomics.

---

## 3. Defects

### 🔴 D1 (Critical): CI eval job's skip condition uses the `secrets` context in a job-level `if` — the gate likely never runs

`.github/workflows/eval-gate.yml:116`:

```yaml
if: github.repository == 'ShreytamGoyal/production-rag' && secrets.NVIDIA_API_KEY != ''
```

Per GitHub's contexts-availability reference, `jobs.<job_id>.if` supports only the
`github`, `needs`, `vars`, and `inputs` contexts — **`secrets` is not among them**
(docs.github.com → Actions → Contexts reference, table row `jobs.<job_id>.if`). The
documented workaround is to map the secret check to an env var at workflow level and test
`env.*` in the job condition.

Consequences: `secrets.NVIDIA_API_KEY` evaluates as unavailable/empty at that evaluation
point, so the whole expression is false and the `eval` job is **skipped on every run**.
The aggregation job (`eval-gate.yml:283-291`) then treats "skipped" as failure unless the
PR carries the `eval-skip-approved` label — i.e., either every PR is red pending a manual
label, or the team routinely applies the label, in which case **the statistical gate is
dead code and no eval has gated any merge**. (Verify against actual workflow-run history;
the fix is mechanical: move the secret-presence check into a workflow-level
`env: HAS_NVIDIA_KEY: ${{ secrets.NVIDIA_API_KEY != '' }}` and gate on that.)

### 🟠 D2 (High): Baseline provenance was lost in the Langfuse migration

The retired file-based runner recorded `git_sha`, `timestamp`, and `model_versions` in each
artifact (`eval/runs/hotpotqa.baseline.results.json:4-6`). The Langfuse-native path records
**none of these**: `run_experiment()` is called with no metadata describing code version,
models, subset size/seed, or item identity (`_langfuse_backend.py:110-114`), and the CI
overrides the sha-bearing default run name (`experiment.py:81`,
`f"{version}@{_git_sha()}-{ts}"`) with `"ci-${{ github.run_id }}"` (`eval-gate.yml:232`),
while the comparison target is just `"baseline"` (`eval-gate.yml:243`). The baseline run is
therefore an **unversioned, unattributed artifact on a remote server**: when it was made,
from what commit, against which models — unknowable from the repo. As code and NIM model
endpoints drift, the gate silently compares against an increasingly stale reference.
The SP5 design explicitly planned provenance ("records fast_subset n/seed, judge_votes,
model versions … for auditability", sp5 spec §5.6) but this did not survive the rewrite.

### 🟠 D3 (High): Silent `0.0` on LLM/parse failures biases the gate toward false regressions

Every LLM-judged metric conflates "computation failed" with "score is zero":

- `faithfulness`: empty claim extraction → `0.0` (`generation_metrics.py:140-142`);
  empty verdict list → `0.0` (`:160-162`);
- `answer_relevancy`: no reverse-questions generated → `0.0` (`generation_metrics.py:194-196`);
- `context_recall`: empty statements/verdicts → `0.0` (`generation_metrics.py:287-288, 306-308`);
- holistic judge: parse failure → `parsed.get("score", 0.0)` (`llm_judge.py:88-92`).

A transient judge outage across a run reads as a catastrophic regression (all-zero means),
and intermittent single-item failures inject downward noise. There is no error sentinel,
no NaN propagation, and no evaluator-level retry (only HTTP-level retries inside the
provider client). Contrast `ragas_adapter.py:88-95`, which correctly emits `float("nan")`
on metric failure and excludes it from aggregates — the native spine lacks the same
discipline. Note also the inverse hazard: terse HotpotQA answers ("Yes") can legitimately
extract zero atomic claims → scored 0 faithfulness for a *correct* answer.

### 🟡 D4 (Medium): One missing score on one item aborts the entire gate with a traceback

`get_run_scores` emits a `RunItemScores` with an **empty dict** for any item whose trace
carries no numeric scores (`_langfuse_backend.py:135-138`). `_aligned_values` then raises
`ValueError("metric 'x' missing on item 'y'")` if *any* of the union metrics is absent on
*any* item (`gate.py:39-40`). So a single flaky evaluator call on 1 of 15 items kills CI
with an exception instead of a per-metric verdict table — loud, yes, but brittle and
uninformative; there is no partial report and no configured tolerance for score gaps.

### 🟡 D5 (Medium): `context_precision` deviates from RAGAS — judged against the model's own answer

RAGAS judges context relevance against the question + **reference** (ground truth). Here
the evaluator passes the pipeline's *generated* answer as "Expected answer"
(`evaluators.py:40` → `generation_metrics.py:230-236`). This is self-referential: contexts
that confidently supported a hallucinated answer get judged "relevant," inflating
context_precision exactly where the system is most wrong, and making CP positively
correlated with generation errors instead of an independent retrieval-quality signal.

### 🟡 D6 (Medium): doc-id / chunk-id semantic conflation — "chunk-level" recall is actually doc-level

- The pipeline exports `retrieved_ids = metadata["retrieved_doc_ids"]` (`pipeline.py:264, 308`),
  while `retrieved_chunk_ids` is available but unused by evaluators.
- The retrieval evaluator compares that doc-id list against `item.relevant_chunk_ids`
  (`evaluators.py:21-27`).
- The ingest exporter writes `relevant_doc_ids` under the JSON key
  `"relevant_chunk_ids"` (`ingest/run.py:191`).

Today this is accidentally consistent because HotpotQA docs are one paragraph ≈ one chunk
and gold ids are Wikipedia titles (`data/eval/hotpotqa.json:5-8`). But the naming is wrong
end-to-end, and the moment arxiv/financebench corpora are chunked into multiple chunks per
doc, the evaluator will keep measuring doc-level overlap while reporting metric names that
say chunk-level — silently mismeasuring retrieval granularity.

### 🟡 D7 (Medium): Subset pairing depends on unrecorded, unordered dataset reads

CI ingests 50 items, then `--fast` samples 15 by *position* from whatever order
`get_dataset_items` returns (`experiment.py:45-52`; `fast_subset.py:19-20`). Langfuse does
not contractually guarantee stable item ordering across re-seeds/UI edits, and nothing
records *which* 15 ids a given run used (D2). If ordering shifts between the day the
`baseline` run was bootstrapped and today's PR run, the gate fails only via the loud
item-set-mismatch exception (`gate.py:25-28`) — correct behavior, but with no recorded
subset identity, diagnosing it requires manual Langfuse archaeology. The SP5 design's
"baseline must match the gated subset" enforcement (spec §5.6) regressed to convention-only.

### 🟡 D8 (Medium): No multiplicity control across nine simultaneously-gated metrics

The gate applies a ~95%-confidence one-sided rule independently to up to 9 metrics
(`gate.py:69-86`; metrics enumerated in `evaluators.py:23-49`). Under a true no-change null
with independent metrics, family-wise false-failure probability approaches
1 − 0.95⁹ ≈ 37%. With n=15 items and noisy judge-based scores (highly correlated metrics in
practice, so real inflation is smaller but material), expect flaky red gates. Combined with
D3's failure-mode zeros, transient provider issues become "regressions."

### ⚪ D9 (Low): Judge-vote mechanics edge cases

Even `judge_votes` takes the upper-middle vote rather than a true median
(`results[len(results)//2]`, `llm_judge.py:96-97`) — votes should be constrained odd. Only
the median vote's rationale survives (`llm_judge.py:97`); the other votes' rationales/scores
are discarded, losing observability the Langfuse UI could otherwise show. And since votes
run at temperature 0.0 with different seeds (`openai_compatible.py:52-56`), providers that
honor neither seed nor sampling variance will return near-identical votes — cost without
variance reduction.

### ⚪ D10 (Low): Documentation and artifact hygiene drift

- `docs/architecture.md:63,300` calls `fast_subset` "stratified"; it is a plain seeded
  random sample (`fast_subset.py:11-20`).
- `docs/architecture.md:397` budgets "Holistic Judge | 1 gen" per item while default
  `judge_votes=3` (`config.py:160`) → actual is 3 gen calls/item.
- Stale `__pycache__` bytecode for removed modules (`eval/__pycache__/run_eval.*`,
  `compare.*`) and git-ignored June-era artifacts in `eval/runs/` (n_items=5,
  `git_sha: "unknown"`) invite confusion about what "the baseline" is — it is now a
  Langfuse run, not these files.

### ⚪ D11 (Low): Retrieval metric conventions

`precision_at_k` divides by `k` even when fewer than `k` results were retrieved
(`retrieval_metrics.py:20-22`) — defensible, but differs from the sklearn/min(k, |retrieved|)
convention; worth documenting. `mrr()` computes reciprocal rank for a *single* ranked list
(`retrieval_metrics.py:40-48`) despite the "mean" name (averaging happens upstream).

---

## 4. Risks

**R1. Single-dataset gate.** Only `hotpotqa` is gated (`eval-gate.yml:228-243`). The
financebench and arxiv adapters exist but are not wired into CI; arxiv's golden set is still
aspirational ("~20 human-verified label pairs should be added … as a separate golden.jsonl",
`corpora/arxiv/adapter.py:11-13`) and its LLM-synthesized items are explicitly *not*
human-verified (`adapter.py:111-113`). A gate on one 15-item slice of one benchmark is a
narrow quality signal for a "production" RAG.

**R2. Eval measures a configuration that differs from production.** Guardrails OFF and
cache OFF during eval (`experiment.py:83-85`): defensible for metric cleanliness, but it
means faithfulness/judge scores characterize *unguarded* generation, while prod answers
pass output guardrails/redaction (including refusal scrubbing at `pipeline.py:271-287`).
Quality regressions introduced by guardrails themselves are invisible to the gate.

**R3. Refusals are penalized, not handled.** When the pipeline legitimately refuses
(`refused=true`, e.g., context insufficient), the refusal text typically yields zero
extractable claims → faithfulness scored 0 (D3 mechanics). HotpotQA is always-answerable so
this is latent today, but the promoted-from-production items that R8 envisions will include
unanswerables — the metric suite has no refusal-aware handling (contrast: judge rubric also
gives no credit dimension for appropriate abstention, `llm_judge.py:16-29`).

**R4. Self-judging bias.** In CI the generator and the judge are the same model —
`meta/llama-3.3-70b-instruct` for both `GEN_MODEL` and `JUDGE_MODEL`
(`eval-gate.yml:157,162`). Same-model-family judging is a documented self-preference risk;
scores are useful relatively (paired deltas) but their absolute levels should not be
interpreted as ground truth. The rubric asks the model to internally average three
dimensions into one score without emitting sub-scores (`llm_judge.py:16-29`), so dimension
attribution is unverifiable.

**R5. Hosted Langfuse is on the merge-critical path.** Dataset reads, experiment writes,
and score read-backs all require the SaaS to be up and correctly permissioned; there is no
offline fallback for gating (the file-based path was removed deliberately,
decomposition-d spec §"Old file-based run_eval/compare.py → Removed"). A Langfuse outage or
quota exhaustion fails the eval job — availability coupling worth an explicit runbook entry.

**R6. Metric scale vs. fixed tolerance.** Historical aggregates show answer_relevancy means
≈ 0.22 with tight CIs (`eval/runs/hotpotqa.baseline.results.json:34-38`): mean cosine of
embedded reverse-questions compresses into a narrow band where the absolute
`eval_tolerance=0.03` is large relative to typical movement, while binary retrieval metrics
on 15 items move in coarse steps (1/15 ≈ 0.067 > tolerance). One tolerance across metrics
of different scales/variabilities is crude; the bootstrap mode mitigates this, threshold
mode does not (floors must be hand-tuned per metric).

**R7. Cross-check adapter can green-light silently.** `ragas_adapter.main()` writes a JSON
summary and exits 0 even when every metric errored to NaN (means become NaN,
`ragas_adapter.py:154-163`). As a human-run validation tool that's tolerable, but nothing
flags an all-NaN cross-check as a failure of the validation exercise itself.

**R8. Trace-promoted dataset items start empty.** `add_item_from_trace` creates items with
empty input/expected_output (`_langfuse_backend.py:60-69`) pending human labeling; if such
items enter a gated dataset before labeling, faithfulness/context_recall compute against
empty ground truth (statement extraction on "" → likely 0 claims → 0.0 per D3) and the item
set includes junk. There is no guard distinguishing labeled from unlabeled items in
`get_dataset_items`.

---

## 5. Recommendations (prioritized)

1. **Fix D1 now**: workflow-level `env:` indirection for the secret check; then confirm in
   Actions history that the eval job has actually been executing. If it never ran, treat
   every historical "gate passed" claim as unverified.
2. **Attach provenance to every run** (D2): pass git sha, model ids, `fast_n/fast_seed`,
   sampled item ids, and timestamp via `run_experiment(...)` metadata (or the dataset-run
   description) in `_langfuse_backend.run_experiment`; have `gate.py` print and sanity-check
   baseline-vs-new metadata before comparing.
3. **Introduce NaN/error sentinels in evaluators** (D3/D4): evaluator exceptions and parse
   failures → `float("nan")` + counted errors; add bounded retries around judge/metric LLM
   calls; make the gate report per-metric coverage and fail loudly when error-rate exceeds a
   configured ceiling instead of reading zeros.
4. **Make `context_precision` reference-based** (D5): pass `item.expected_output`, matching
   RAGAS semantics and de-correlating it from generation quality.
5. **Resolve the id vocabulary** (D6): rename `relevant_chunk_ids`→`relevant_doc_ids`
   end-to-end now, and switch the evaluator to true chunk-level ids once any corpus chunks
   documents.
6. **Record subset identity** (D7): persist the sampled item-id list (or its hash) per run;
   optionally sort items deterministically before sampling.
7. **Tame multiplicity** (D8): Holm correction across metrics, or gate on a small pre-declared
   set (e.g., recall_at_5, faithfulness, judge_score) and merely *report* the rest.
8. **Wire financebench (and later a verified arxiv golden.jsonl) as second/third gate datasets** (R1).
9. **Judge improvements** (D9/R3/R4): enforce odd votes; push each vote's score/rationale as
   separate Langfuse scores; special-case refusals in metrics; consider a judge model
   different from the generator family.
10. **Hygiene** (D10): correct the "stratified" and judge-budget doc claims; delete stale
    `__pycache__` bytecode for removed modules and stale `eval/runs/` artifacts (or move
    them under a clearly-named `archive/`).

---

## 6. Verdict

The framework's statistical core and software architecture are genuinely strong — the
paired-bootstrap non-inferiority gate, strict item alignment, dependency-injected offline-
testable metrics, and seeded reproducibility are best-practice and clearly hardened against
specific earlier defects. The two most serious problems sit *around* that core: the CI
condition that likely prevents the gate from ever executing (D1), and the loss of baseline
provenance plus silent-failure metric semantics (D2/D3), which together undermine how much
a green gate actually certifies. Fixing D1–D4 is cheap relative to the credibility they
restore to the release process.
