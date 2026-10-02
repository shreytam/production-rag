# P0 — API ingest path: collection creation + dead chunking config

Source: `docs/BACKLOG.md` § "P0 — bugs found but not yet fixed".

Two independent P0 defects on the **API** ingest path (`app/documents.py` → arq →
`ingest/worker.py` → `IncrementalIngestor` → `QdrantVectorStore.upsert`). Both are
production-wiring bugs that the test fixtures currently paper over.

## Global Constraints

- **TDD.** Write the failing test first, watch it fail for the right reason, then
  implement. Every task ships with tests that fail before the change and pass after.
- **Fail-closed.** The worker's existing contract stands: any error marks the
  document `failed` (`ingest/worker.py` `except Exception`) — never a partial `ready`.
- **No compat shims.** This project was rebuilt from scratch and has no external
  consumers. Prefer a clean change over a backwards-compatible alias.
- **Do not weaken existing tests to make new code pass.** If a fixture must change,
  the change must make the fixture *more* faithful to production, not less.
- **Commit authorship:** commits are authored solely as
  `Shreytam Goyal <shreytamgoyal@gmail.com>`. No `Co-Authored-By:` trailer, no
  "Generated with Claude" note, no AI/Anthropic attribution of any kind in commit
  messages. This is a hard repository rule (`CLAUDE.md`).
- Run the full suite (`.venv/bin/python -m pytest -p no:warnings --tb=no -q`;
  **exit code 0 is the pass signal** — this repo's pytest config suppresses the
  final summary line) and `.venv/bin/ruff check .` before reporting DONE.
- Baseline at branch point: full suite exit 0, ruff clean.

---

## Task 1 — Call `ensure_collection()` on the API ingest path

### The bug

`ingest/run.py:163` (the CLI/eval path) is the **only** caller of
`vector_store.ensure_collection(...)` in non-test code. The API path never creates
the Qdrant collection. Point a fresh Qdrant at the API and the first upload upserts
into a collection that does not exist; the document lands in `failed`.

No test catches it today because both fake vector stores stub `ensure_collection`
out as a no-op *and* accept `upsert` unconditionally:
- `tests/_fakes.py:44`
- `tests/test_pipeline_integration.py:54`

### Requirements

1. The API ingest path must ensure the Qdrant collection exists before the first
   `upsert`. Call it **once at worker startup** — not per document, since it is two
   round trips (`get_collections` + `create_collection`) plus two
   `create_payload_index` calls. A lazy once-per-process guard on first upsert is an
   acceptable alternative if you justify it in the report; per-document is not.
2. The dimension argument is `embedder.dimension` — the same value `ingest/run.py:163`
   passes. Source it from the embedder the worker already builds; do not hardcode it
   and do not read it from a second config knob.
3. Startup must be fail-closed and observable: if `ensure_collection` raises at
   worker startup, the worker must not silently proceed to accept jobs.
4. **A test must be able to catch this regression.** Make at least one fake vector
   store faithful: it must record whether `ensure_collection` was called and **reject
   `upsert` when it was not** (raising the way a real Qdrant does for a missing
   collection). Then assert the API/worker path reaches `ready`. Removing the stub
   entirely is fine if the suite stays green.
5. Add a test that drives the **real** `QdrantVectorStore` against an empty Qdrant
   and asserts the first upload reaches `ready`. Follow the existing live-DB test
   convention in `tests/test_stores_acl.py` / `tests/test_multitenant_isolation.py`
   (skip when the DB is unreachable) so it runs in the `acl-isolation` CI job and
   skips locally without Qdrant.
6. Audit the rest of the fixture-stubbed wiring in `app/documents.py` for the same
   class of defect — `get_registry`, `get_blobs`, `get_parsers`. **Report what you
   find; do not fix unrelated findings in this task.** This is the same class as the
   arq `redis_settings` bug fixed in #20: production wiring the fixtures hide.

### Out of scope

Anything in Task 2. Any fix for wiring defects found under requirement 6 — report
them, do not implement them.

---

## Task 2 — Thread chunking configuration through both ingest paths

### The bug

`ingest/worker.py:84` calls `chunk_document(doc)` with no arguments, silently taking
the signature defaults `max_tokens=256, overlap=32` (`ingest/chunking.py:65-69`).
`Settings.chunk_overlap = 200` (`core/config.py:105`) is read by **nothing** outside
`tests/test_sp4_config.py`. Chunk size is not configurable at all, on any path.

### Requirements

1. Add `chunk_max_tokens: int = 256` to `Settings` (`core/config.py`), beside the
   existing `chunk_overlap`.
2. **`chunk_overlap` keeps its declared default of `200`, and that default now takes
   effect.** This is a deliberate, human-made decision: the config value as written
   wins over the value currently in force. Do **not** "fix" the discrepancy by
   lowering the default to 32 — honouring 200 is the point of the task.
3. Thread both knobs through **both** ingest paths — `ingest/worker.py` (API) and
   `ingest/run.py` (CLI/eval) — so a config change actually changes chunking on
   every path. Neither path may keep calling `chunk_document(doc)` bare.
4. `chunk_document`'s own signature defaults stay as they are; configuration is
   supplied by the callers, not by mutating the function's defaults.
5. Validate the relationship between the two knobs at boot. `core/config.py` already
   carries `@model_validator(mode="after")` validators — follow that pattern. An
   `overlap >= max_tokens` configuration cannot chunk and must not boot.
   **Note the live values: `overlap=200` against `max_tokens=256` is legal but is
   ~78% duplication.** Verify the chunker actually terminates and produces sane
   output at these values — if it does not, that is a real finding: report it as
   `DONE_WITH_CONCERNS` rather than quietly adjusting the defaults.
6. Tests: assert that a non-default `chunk_max_tokens` / `chunk_overlap` in
   `Settings` demonstrably changes the chunks produced by **each** ingest path.
   A test that only asserts the setting exists does not satisfy this.

### Known consequence (accepted, do not re-litigate)

Honouring `overlap=200` changes retrieval behaviour — chunks will overlap heavily.
The human partner accepted this explicitly. There is no committed eval baseline to
invalidate (`docs/BACKLOG.md` § P3 records that the `baseline` Langfuse dataset run
does not exist yet), so nothing is being silently regressed against. Call the
behaviour change out in the commit message.

### Out of scope

Anything in Task 1. Re-running or creating an eval baseline. Any change to
`chunk_document`'s chunking algorithm itself.
