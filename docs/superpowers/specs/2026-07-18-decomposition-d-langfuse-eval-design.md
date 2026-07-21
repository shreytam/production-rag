# Decomposition D — Langfuse-native Evaluation (Design)

Date: 2026-07-18
Status: Approved (design); implementation plan to follow.

## Purpose

Move the RAG evaluation workflow onto **Langfuse Datasets**. Two goals, weighted
equally:

1. **Managed experiment tracking** — every eval run is a Langfuse dataset run
   ("experiment"): per-item outputs and metric scores land in the Langfuse UI,
   linked to their pipeline traces, so versions/models can be compared over time.
2. **Trace-driven dataset curation** — golden datasets are built and grown inside
   Langfuse, including promoting real production traces into dataset items.

This is **Approach B**: Langfuse is the authoritative, server-hosted home for the
dataset and for every run. The eval loop is driven by the Langfuse SDK's
`run_experiment(...)`. The user hosts a persistent Langfuse server; CI reaches it.

### Accepted trade-off

The PR **eval gate now hard-depends on the hosted Langfuse being reachable** with
valid CI credentials. This is inherent to B. The **offline lint/test job does not**
depend on Langfuse — all eval modules import `langfuse` lazily and are unit-tested
against a fake backend.

## Scope

**In scope (integration plumbing, unit-tested with a fake Langfuse client):**

- An `EvalBackend` seam wrapping the Langfuse SDK (the only place `langfuse` is
  imported).
- A Langfuse-native experiment runner (`experiment.py`) built on `run_experiment`.
- A regression gate (`gate.py`) that fetches run scores from Langfuse and applies
  a configurable pass/fail policy.
- A curation CLI (`dataset_cli.py`): `seed` and `add-from-trace`.
- Config additions and CI wiring for the online eval job.

**Out of scope (explicit):**

- Curating real datasets or running a live NIM baseline. No golden dataset or
  baseline run is produced in this decomposition; that is a separate follow-up.
- Langfuse server-side evaluators. Metrics are computed **client-side** and pushed
  as scores (retrieval metrics require `relevant_chunk_ids` comparison and cannot
  be expressed as server evaluators anyway).
- `eval/ragas_adapter.py` — left untouched; remains an optional cross-check.
- The Redis semantic cache (its own deferred plan).

## Decisions (locked)

| Decision | Choice |
|---|---|
| Goal | Experiment tracking AND trace-curation, equally |
| Architecture | B — Langfuse-native; dataset + runs live on the server |
| Runner driver | Langfuse SDK `run_experiment(task, evaluators, ...)` |
| Metric scoring | Client-side compute, pushed as Langfuse scores |
| Gate mechanism | Configurable: paired-bootstrap-vs-baseline (default), absolute thresholds, or both |
| Curation | `add-from-trace` CLI + native Langfuse UI button |
| Dataset/baseline population | Out of scope; plumbing only, tested with fakes |
| Old file-based `run_eval.py`/`compare.py` | Removed (clean rename, no compat shim) |

## SDK contract (Langfuse 4.9.1, verified)

- `Langfuse.get_dataset(name) -> DatasetClient` — `.items`, each item exposing
  `.input`, `.expected_output`, `.metadata`.
- `Langfuse.run_experiment(*, name, run_name=None, data, task, evaluators=[],
  run_evaluators=[], max_concurrency=50, metadata=None) -> ExperimentResult` —
  drives the loop, auto-links a trace per item, creates the dataset run, and
  attaches evaluator outputs as scores.
- `TaskFunction`: `(*, item, **kwargs) -> output`.
- `EvaluatorFunction`: `(*, input, output, expected_output, metadata, **kwargs)
  -> Evaluation | list[Evaluation]` (`Evaluation` carries name/value/comment).
- `Langfuse.get_dataset_run(*, dataset_name, run_name) -> DatasetRunWithItems` —
  per-item results and scores; used by the gate.
- `Langfuse.create_dataset(*, name, ...)` and
  `Langfuse.create_dataset_item(*, dataset_name, input, expected_output, metadata,
  source_trace_id=None, ...)` — `source_trace_id` gives native provenance for
  `add-from-trace`.

## Architecture

Eval is a Langfuse experiment. Existing metric code is reused verbatim, re-wrapped
as evaluators. A single `EvalBackend` protocol isolates the SDK so the runner,
gate, and CLI stay unit-testable offline.

### Module map (`eval/`)

| Module | Fate | Purpose |
|---|---|---|
| `langfuse_eval.py` | new | `EvalBackend` protocol + real SDK wrapper. The only module importing `langfuse`. |
| `experiment.py` | new (replaces `run_eval.py`) | Builds `task` + `evaluators`, calls `run_experiment`. |
| `gate.py` | new (replaces `compare.py`) | Fetches new-run + baseline-run scores, applies the gate. |
| `dataset_cli.py` | new | `seed` (bootstrap dataset from local items) + `add-from-trace <trace_id>`. |
| `retrieval_metrics.py`, `generation_metrics.py`, `llm_judge.py`, `stats.py`, `fast_subset.py` | kept, unchanged | Metric math, bootstrap, item subsetting — reused. |
| `run_eval.py`, `compare.py` | removed | Superseded by `experiment.py` / `gate.py`. |
| `ragas_adapter.py` | untouched | Out of scope. |

## Data model

A golden item maps to a Langfuse dataset item:

- `input` = `{"question": "<text>"}`
- `expected_output` = ground-truth answer string
- `metadata` = `{"relevant_chunk_ids": [...], "tenant_id": "public"}`

A `run_experiment` call is one dataset run, named `full@<git_sha>` by default (a
CI-supplied `--run-name` overrides). Each item's pipeline execution is an
auto-linked trace; each metric is a score on that run item.

## Components

### EvalBackend seam (`langfuse_eval.py`)

A `Protocol` exposing only what eval needs: `get_dataset`, `run_experiment`,
`get_dataset_run`, `create_dataset`, `create_dataset_item`. The real
implementation wraps a `Langfuse` client, reusing the construction/config already
used by `observability/langfuse_tracing.py`. `langfuse` is imported lazily inside
this module so importing any other eval module needs neither the server nor the
package. `FakeEvalBackend` (in tests) holds datasets/items/runs/scores in memory.

### Runner (`experiment.py`)

Builds and submits an experiment:

- `task(*, item) -> output`: runs
  `pipeline.run(item.input["question"], acl=ACLContext(tenant_id=item.metadata["tenant_id"]))`
  and returns the pipeline result dict (`answer`, `retrieved_ids`, `contexts`, …).
  Guardrails are forced OFF (as today) to avoid confounding metrics.
- `evaluators`: adapters matching `EvaluatorFunction`, delegating to existing
  metric functions:
  - one retrieval evaluator → `recall@5`, `precision@5`, `mrr`, `ndcg@5`
    (from `output.retrieved_ids` vs `metadata.relevant_chunk_ids`);
  - generation evaluators → `faithfulness`, `answer_relevancy`,
    `context_precision`, `context_recall`;
  - holistic judge → `judge_score`.
  Generator/embedder are built once (from `core.registry`) and closed over.
- Calls `backend.run_experiment(name=<dataset>, run_name=<name>, data=<items>,
  evaluators=[...], max_concurrency=<config>)`. `--limit` and `--fast` (reusing
  `fast_subset`) cap item count for CI.

### Gate (`gate.py`)

`get_dataset_run` for both the new run and the configured baseline run → reshape
scores to `{metric: [per-item values]}` → apply, per `eval_gate_mode`:

- `bootstrap` (default): paired bootstrap vs baseline (reuses
  `stats.paired_bootstrap`); fail if the regression confidence bound exceeds
  `eval_tolerance` — identical logic to the current `compare.py`.
- `threshold`: each metric mean must clear its floor in `eval_gate_thresholds`.
- `both`: run both; fail if either fails.

Exits nonzero on failure and prints a metrics table.

### Curation CLI (`dataset_cli.py`)

- `seed --dataset <name> --items <path>`: creates the dataset if absent and
  upserts hand-authored items idempotently (bootstrapping + tests).
- `add-from-trace --dataset <name> --trace-id <id>`: fetches the trace and calls
  `create_dataset_item(source_trace_id=<id>, input=<question>, expected_output="")`;
  a human fills the expected output in the UI. The native "add to dataset" UI
  button is the documented manual complement.

## Data flow

1. **Curate:** prod trace → (`add-from-trace` CLI or UI button) → dataset item.
2. **Run:** `experiment.py` → `run_experiment(task, evaluators)` → per-item linked
   traces + scores + a named run on the server.
3. **Gate:** `gate.py` → fetch new + baseline runs → bootstrap/threshold → pass/fail.

## Configuration & CI

### Config (`core/config.py`)

Add: `langfuse_host`; `eval_gate_mode` (`bootstrap|threshold|both`, default
`bootstrap`); `eval_gate_thresholds` (metric→floor map); `eval_baseline_run` (run
name used as the gate baseline). Reuse existing `eval_tolerance`, judge/bootstrap
knobs, and the existing Langfuse public/secret key settings.

### CI (`.github/workflows/eval-gate.yml`)

- Online `eval` job gains `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` secrets. Steps become: ingest fixture →
  `python -m eval.experiment` → `python -m eval.gate`.
- Fix the stale embedding pin: `nv-embedqa-e5-v5` → `bge-m3` (aligns CI with the
  current default).
- The offline lint/test job is unchanged (fakes; no Langfuse, no secrets).

## Testing (offline, `FakeEvalBackend`)

- Runner: `task` builds the expected output; the expected score set is emitted per
  item; `--limit`/`--fast` honored.
- Evaluators: exact values on known inputs (retrieval math exact; generation/judge
  via a stub generator/embedder).
- Gate: `bootstrap` fails on an injected regression and passes within tolerance;
  `threshold` fails below a floor; `both` composes correctly.
- CLI: `add-from-trace` produces an item with `source_trace_id` set;
  `seed` upserts idempotently.

## Error handling & isolation

- Missing/empty dataset, baseline run not found, or Langfuse unreachable → fail
  loudly with an actionable message. Eval is a hard dependency under B; it never
  silently passes.
- A run with incomplete scores → the gate errors rather than gating on partial
  data.
- `langfuse` is imported lazily inside the real backend only. Importing any eval
  module — and therefore lint and the offline test suite — requires no server and
  no `langfuse` package, mirroring the guarded arq import in the ingest worker.

## Success criteria

- `experiment.py`, `gate.py`, `dataset_cli.py`, and `langfuse_eval.py` exist with
  the interfaces above; `run_eval.py` and `compare.py` are removed.
- Full offline test suite green with no Langfuse server or package required for
  collection.
- Gate logic (bootstrap/threshold/both) is unit-tested against `FakeEvalBackend`.
- CI eval job wired to the hosted Langfuse; offline lint/test job unaffected.
- No live dataset or baseline is produced (out of scope, by design).
