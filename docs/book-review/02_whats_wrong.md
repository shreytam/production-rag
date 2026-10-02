# Artifact 2 — What's Wrong: Defects, Risks & Gaps

> Companion to `01_whats_right.md` and `03_suggestions.md`. Every finding was verified against source at branch `ui-test-console` (@ `3af99ff`) by four parallel audits plus first-hand reads. Full evidence lives in `docs/review/audit_*.md`.

## The five things to know up front

1. **You have a live NVIDIA API key sitting in a file nothing ignores** (`infra/.env.bak-1785181911`). One `git add -A` or one `docker build` away from leaking. Rotate it today.
2. **Your CI does not run at all.** `.github/workflows/eval-gate.yml:116` uses the `secrets` context in a job-level `if:`, which GitHub Actions rejects — the whole workflow fails to parse. The statistically rigorous eval gate you built currently protects nothing.
3. **The semantic cache can serve user A's answer to user B** once ACL tags enter real use — cache identity omits the caller's tag set (`cache/semantic_cache.py:27-35`). Latent only because ingestion happens to write zero tags today.
4. **Retrieval degrades silently instead of loudly**: a NIM reranker blip returns `[]` and generation confidently answers from zero context; CLI-ingested corpora get an empty BM25 leg because `ingest/run.py` populates a legacy pickle the production retriever never reads.
5. Several advertised config knobs are **dead** (`chunk_overlap=200` vs actual hardcoded 32; `max_chunks_per_corpus` unused anywhere; `hybrid_require_sparse` self-documented as inert). Operators tuning them change nothing.

---

## 🔴 Critical — act this week

### CR-1 · Live credential exposed in an unignored backup file
- **Where:** `infra/.env.bak-1785181911` — contains a real 70-char `nvapi-` key. Also: `docs/PRODUCTION_READINESS_AUDIT.md` prints a full key verbatim in its finding text (that doc is itself a leak artifact).
- **Why it matters:** `.gitignore` covers `.env`/`.env.local` and `.dockerignore` covers `.env`, `.env.local`, `infra/.env` — **neither matches `*.bak*`**. The Dockerfile's `COPY . .` bakes it into image layers; `git add -A` stages it. `PROJECT_STATUS.md` §8 claims this leak class is closed — it re-opened on 2026-07-28 when the backup was created.
- **Fix:** rotate the key now → delete both artifacts → add `infra/.env*` and `*.bak*` to both ignore files → add gitleaks to pre-commit + CI.

### CR-2 · CI is structurally broken — no job ever runs
- **Where:** `.github/workflows/eval-gate.yml:116`: `if: github.repository == '…' && secrets.NVIDIA_API_KEY != ''`. The `secrets` context is not available in `jobs.<job_id>.if` (GitHub docs context-availability table; actions/runner#520).
- **Impact:** the workflow file fails validation → lint, offline tests, ACL-isolation tests, eval gate, and the status gate **never execute**. Worse than "gate awaiting baseline": there is no CI whatsoever, so even the safety nets that don't need secrets aren't running.
- **Fix:** drop `secrets.*` from the job-level `if` (keep the repository check); detect key presence via a workflow-level `env:` mapping or a check job with outputs; run `actionlint` locally and in CI to catch regressions. Two independent audits found the same defect — this is certain, not speculative.

### CR-3 · Semantic cache ignores caller ACL tags → within-tenant data disclosure (latent)
- **Where:** `cache/semantic_cache.py:27-35` keys lookup/store on `(tenant_id, collection_id, embedding)` only. `core/pipeline.py:160-169` returns cached Answers whose contexts/citations were produced under *another principal's* authorization.
- **Scenario:** user A (tag `hr`) asks Q → answer citing HR-only chunks cached under `(tenant, collection, vec)`. User B (no tags, same tenant) asks something semantically similar → served A's answer without any ACL check against those chunks. Your store-level ACL enforcement — excellent everywhere else — is bypassed wholesale above the stores.
- **Latency of the bomb:** current ingestion paths all write empty `acl_tags` (`ingest/worker.py:77`, `ingest/base.py:55`), so it hasn't fired. But JWT claims already carry tags and the schema advertises them — enabling the feature arms this.
- **Fix:** include a digest of the caller's sorted `acl_tags` in both cache tiers' identity (or partition per tag-set). Must land together with CR-4 (see sequencing note there).

---

## 🟠 High — correctness and security defects

### HG-1 · Re-ingest doesn't propagate tightened ACLs to text-unchanged chunks
`ingest/incremental.py:14-16` detects tag changes via `_meta_hash` (it includes `acl_tags`) — but the metadata-update path writes only `{"title": c.title}` (`incremental.py:50-51`). Qdrant patches exactly that dict (`qdrant_store.py:164-173`), so `acl_tags`/`acl_open` stay stale. Narrowing a document's visibility leaves old broad visibility on every unchanged point — security drift with a green "re-ingest succeeded" status. Edge case: changing a doc's tenant_id orphans all old-tenant points forever.
**Fix:** on meta-hash mismatch, upsert the full ACL payload (`acl_tags`, `acl_open`, `collection_id`); treat tenant change as delete+create. Land in the same change-set as CR-3 — fixing the cache alone would just surface stale tagged payloads here.

### HG-2 · Reranker outage degrades into wrong answers, not degraded ranking
`nim_rerank.py:62` propagates HTTP errors (only timeouts/network retry); `HybridRetriever.retrieve` has no try/except (`hybrid.py:57-61`). Worse: a 200-with-empty-`rankings` response normalizes to `[]` (`rerankers/_common.py:16-29`) → fused candidates discarded → `retrieve()` returns `[]` → the generator answers "I cannot answer" fleet-wide while dense+BM25 results were fine. No flag, no metric, no fallback.
**Fix (small, do early):** wrap `reranker.rerank`; on exception *or* implausibly-empty output vs non-empty input, return the pre-rerank RRF window and set `degraded=true` in response metadata.

### HG-3 · Sparse-index cache breaks read-your-writes across processes
`TenantSparseStore._retriever` memoizes per process and reloads only if the tenant is absent from the dict (`sparse/tenant_store.py:31-40`). The API process builds its pipeline once at startup and keeps its BM25 indexes forever; the arq worker writes via atomic file replace in another process. Result: after every ingest/update/delete, previously-queried tenants retrieve deleted/stale chunks over BM25 until an API restart — new docs invisible to the sparse leg, deleted docs still retrievable. Dense stays fresh, so RRF fuses two epochs of the corpus. Bonus race: `_save` uses a fixed `<hash>.tmp` name, so two writers can interleave into a partial pickle (`tenant_store.py:45`).
**Fix:** stat/mtime check before using a cached index (or pub/sub invalidation or TTL); unique tmp names.

### HG-4 · CLI ingest feeds a sparse index the production path never reads
`ingest/run.py:167-175` pickles a `BM25Retriever` to `.cache/bm25_{dataset}_{store}.pkl`. Production resolves sparse exclusively through `TenantSparseStore` (`core/pipeline.py:347-351`); the pickle loader's sole caller is a test. Anything ingested via `python -m ingest.run` runs hybrid retrieval with a **silently empty BM25 leg** — RRF quietly degenerates to dense-only, no error, no metric. Retrieval-quality work would be measured on a pipeline that isn't the one serving traffic.
**Fix:** route `run.py` through `IncrementalIngestor`/`TenantSparseStore` like the worker; delete or fence the pickle path (also retires HG-5 below).

### HG-5 · Unauthenticated pickle deserialization = RCE surface
`pickle.loads` on index files (`tenant_store.py:37`, `run.py:172-174`, `pickle_loader.py:24-26`). The isinstance checks run *after* `pickle.load` — i.e., after arbitrary code execution. Any write primitive or tampered volume owns the API/worker process. No HMAC/signature.
**Fix:** switch snapshots to JSONL of Chunk models (cheap, safe) or HMAC-sign payloads with a server-side key.

### HG-6 · JWKS refetch has no cooldown — unauthenticated DoS amplification
Every unknown `kid` triggers a fresh JWKS fetch with no rate/cooldown (`providers/auth/jwt_verifier.py:49-96`). An attacker sending garbage-kid JWTs turns your verifier into a fetch flood against your IdP (and hammers yourself).
**Fix:** cooldown/backoff per kid, short JWKS TTL cache, cap fetches per interval.

### HG-7 · Groundedness fail-open becomes a permanent silent bypass under load
The design is right (judge timeout shouldn't block traffic), but the shared 4-worker pool starves when judge calls are slow (`output_groundedness.py` + default pool), after which **every** answer passes unverified with only a per-item `groundedness_unverified` marker nobody aggregates. Combined with the 600 s × 5-retry upstream ceiling, a slow judge day = zero grounding verification, invisibly.
**Fix:** dedicated bounded executor + queue-depth circuit breaker; alert when the soft-fail rate crosses a threshold.

### HG-8 · No rate limiting, concurrency bound, or request deadline — denial-of-wallet ready
`app/api.py` registers zero middleware; the only input bound is question length (8000 chars). Each `/query` performs paid embed→search→rerank→LLM work with a 600 s ceiling and 5 retries (`core/config.py:87-88`); sync handlers share a ~40-thread pool. ~40 concurrent slow queries pin the entire service; every request is unmetered inference spend. The book calls this exact threat "denial-of-wallet" and prescribes rate limiting + budget alerts (Ch. 4).
**Fix:** token-bucket middleware (per principal + IP), bounded semaphore around pipeline calls, per-request deadline far below 600 s for interactive use.

### HG-9 · The containerized `api` service cannot serve traffic as shipped
Compose gives `api` only `REDIS_URL/QDRANT_URL/PG_DSN/DOC_REGISTRY_BACKEND` — no `env_file:` — and the image deliberately ships no `.env`. Inside the container: `JWT_SECRET=""` → every request 401s; provider keys empty → the first embed call fails. Only `ingest-worker` works (bind-mounts the repo and reads `.env` via pydantic-settings). `make up`'s "everything" is misleading.
**Fix:** `env_file: [.env]` (or compose secrets) on `api`; document the required-var contract; smoke-test mint-token → upload → query end-to-end.

---

## 🟡 Medium — quality, consistency, operability

| # | Finding | Evidence | Impact |
|---|---|---|---|
| MD-1 | Eval metrics conflate "LLM/judge failed" with "score 0.0" — a judge outage reads as a catastrophic regression; terse answers ("Yes") legitimately extract zero claims → faithfulness 0 for a correct answer | `generation_metrics.py:140-162,194-196,287-308`; contrast `ragas_adapter.py:88-95` which correctly emits NaN | False-positive gate failures; noise |
| MD-2 | `context_precision` judges contexts against the model's **own generated answer**, not ground truth (self-referential) — inflates precision exactly when the system hallucinates confidently | `evaluators.py:40` → `generation_metrics.py:230-236` | Misleading retrieval-quality signal |
| MD-3 | Baseline provenance lost in Langfuse migration: no git sha/timestamp/model versions recorded; CI renames runs to `ci-${run_id}` and compares against bare `"baseline"` — an unversioned yardstick that drifts | `_langfuse_backend.py:110-114`; `eval-gate.yml:232,243` | Gate may compare against a stale reference unknowably |
| MD-4 | Doc-id/chunk-id conflation: evaluators compare doc ids against `relevant_chunk_ids` goldens; exporter writes doc ids under the key `"relevant_chunk_ids"` (`run.py:191`). Works only while 1 doc = 1 chunk | `pipeline.py:264,308`; `evaluators.py:21-27` | Silent mismeasurement once multi-chunk corpora arrive |
| MD-5 | No multiplicity control: ~95% one-sided rule applied independently to up to 9 metrics → family-wise false-failure up to ~37% under a true null (n=15 items) | `gate.py:69-86` | Flaky red gates erode trust in the gate |
| MD-6 | Worker marks documents FAILED then returns normally — no arq retries/job timeout/dead-letter; one NIM 429 storm permanently fails every in-flight doc; uploads can strand in `processing` forever if enqueue fails (blob written, registry row stuck) | `ingest/worker.py:96-99,117-120`; `app/documents.py:147-163` | Ingest reliability gap |
| MD-7 | Store mutations use caller-*visibility* semantics where ownership semantics are needed: worker acts as no-tag caller → once tagged chunks exist, worker deletes/update silently match nothing (FilterSelector finds no points) | `worker.py:72,111` + `retrieval/acl.py:39-50`; BM25 delete ignores tags entirely (`bm25.py:113-119`) | Ghost deletions once tagging goes live |
| MD-8 | Dead/dishonest config knobs: `chunk_overlap=200` advertised, both call sites hardcode defaults (actual overlap **32**) (`config.py:105` vs `ingest/run.py:116`, `ingest/worker.py:84`); `max_chunks_per_corpus` has zero usages; `hybrid_require_sparse` self-documented inert (`config.py:96-100`) | — | Operators tune knobs that do nothing |
| MD-9 | Qdrant client: no `api_key` support anywhere in Settings (Qdrant Cloud impossible without code change), no client timeout, broad except swallowing all `create_payload_index` errors, no explicit `wait=True` | `qdrant_store.py:71,87-106` | Ops blind spots |
| MD-10 | Embedding dimension trusted, never verified: response vector lengths unchecked vs `embed_dimension`/collection schema; `truncate=END` silently drops long-chunk tails with zero telemetry | `openai_compatible.py:35-50`; `qdrant_store.py:74-84` | Model swaps fail weirdly or corrupt retrieval silently |
| MD-11 | Contextual prefixer: full `doc_text` shipped per-chunk with no length cap (cost O(chunks×doc_len), window blowout on huge docs); non-atomic cache-file reads crash ingest on truncated JSON; detector rebuilt per chunk; cache key omits prompt/model version despite `prompt_version` existing | `contextual.py:51-53,77-118` | Cost spikes + fragile ingest |
| MD-12 | Langfuse sees zero tokens/cost: generic spans only — native `usage_details=`/`cost_details=` never populated; Anthropic path priced at $0.00 (priced against `gen_model` setting, not server-returned id) | `langfuse_tracing.py:186-188`; `pipeline.py:207-213,249-253` | Cost analytics dead even when tracing is on |
| MD-13 | Tracing failure indistinguishable from tracing-off: init/span/flush errors swallowed silently | `langfuse_tracing.py:132-134,185-191` | Operators believe traces flow when none do |
| MD-14 | `/healthz` is liveness-only; no `/readyz`; no Dockerfile HEALTHCHECK; lazy singleton build has an unguarded race on cold start | `api.py:27-34,74-76` | Orchestrators route traffic to broken instances |
| MD-15 | Raw 500s on provider hiccups: no exception mapping (502/504/400), no correlation ID, no structured body | `api.py:92` | Debugging + client trust |
| MD-16 | Naive BM25 tokenizer (whitespace+lowercase) and O(N) `get_scores` per query per tenant; `fuse_window=40` and NIM timeouts hardcoded outside Settings | `bm25.py:22-24,70`; `hybrid.py:40`; `nim_rerank.py:24` | Recall tax + scaling ceiling + config-surface lie |
| MD-17 | Eval breadth/config gaps from the deep eval pass: only hotpotqa is gated (financebench/arxiv adapters unwired, arxiv goldens unverified); eval runs with guardrails+cache OFF so regressions *introduced by guardrails* are invisible to the gate; legitimate refusals score faithfulness 0 (no abstention credit); one fixed `eval_tolerance` across metrics of very different scales; trace-promoted dataset items enter gates before human labeling | `eval-gate.yml:228-243`; `experiment.py:83-85`; `generation_metrics.py`; `_langfuse_backend.py:60-69` | The gate certifies less than it appears to (`audit_eval.md` §4, R1–R8) |

---

## ⚪ Lower severity (worth a cleanup PR)

- Leftover document tail dropped when ≤ overlap tokens (`chunking.py:111-113`) — up to ~32 tokens of every doc never indexed.
- Tokenizer guessing maps all non-GPT models (your Llama-on-NIM default!) to `cl100k_base`, and `context_token_safety_margin=0.0` — budget approximate exactly where you deploy (`context_assembly.py:27-35`).
- Local cross-encoder lazy-loads/HF-downloads on first production query → cold-start spike (`local_cross_encoder.py:32-39`).
- Parser registry exact-matches content types (`text/plain; charset=utf-8` fails); `unstructured.partition.auto` has no timeout/page cap — a 25 MiB scanned PDF can pin a worker for minutes.
- Fresh `httpx.Client` per rerank call; NIM reranker gets raw `chunk.text`, dropping the contextual prefix the other legs indexed.
- Even `judge_votes` takes upper-middle rather than true median; discarded vote rationales lose observability.
- Naive p95 in `dashboard.py:69` (misreports for n≤20).
- Stale cruft: `.claude/worktrees/product-ingestion-api/` duplicates the tree (including the cache code — future greps will hit it); `docs/PRODUCTION_READINESS_AUDIT.md` predates major fixes, still cites removed modules, and leaks a key; PROJECT_STATUS §2 contradicts §8 on trace redaction; README quickstart hand-runs arq with no `make worker` target.
- `pg_dsn` default embeds a placeholder password string inviting copy-paste.

---

## Gaps vs the book — prescribed practices that are absent

These aren't bugs; they're chapters of the playbook you haven't implemented yet:

1. **No streaming responses** (Ch. 3–4 UX): `/query` returns one blocking JSON payload. The book treats streamed tokens + progress explanation ("retrieving → generating") as table stakes for perceived latency — especially since your honest p50 will be seconds on a 70B model.
2. **No user-feedback loop** (Ch. 6 pp. 183–184): no thumbs endpoint, nothing stored backend-side. The book makes satisfaction rate (`👍/(👍+👎)`) the **primary production KPI**, correlated against automated metrics. This is the single highest-leverage missing feature — it's how you find out your proxy metrics are lying.
3. **No metadata-filter query API** (Ch. 2 Table 2-1): filtering is a plain DB operation alongside semantic search, exposed so users can scope ("who=user", date ranges). Your stores filter internally on ACL/collection, but callers can't express content filters.
4. **No diversity stage (MMR)** and **no parent-document/small-to-big retrieval** (Ch. 3 p. 79, Ch. 2): reranking optimizes pure relevance; redundant near-duplicate chunks waste your 4000-token budget. Small-to-big (retrieve 256-token chunks, generate over parent sections) is the standard fix for the recall/cohesion trade-off your fixed 256-token chunks create.
5. **Thin parsing for the formats that matter** (Ch. 2 pp. 28–37: "accuracy at this stage is paramount"): parser registry is plain-text + unstructured-auto. No table-aware PDF extraction (`find_tables`), no OCR triage path, no VLM fallback tier for complex layouts, no boilerplate stripping, no encoding normalization — the exact multistage conditional preprocessing pipeline the book prescribes for dirty enterprise data.
6. **No staging-verification workflow for re-ingestion** (Ch. 4 p. ~108): the book's pattern is ingest → staging collection → automated retrieval unit tests → promote. Your idempotent manifests cover crash-safety, but a bad parse can still pollute the live index for everyone with no pre-promotion check.
7. **Judge independence & calibration** (Ch. 6 pp. 164–167): your judge defaults to the *same* 70B model that generates answers — the book's self-referential-bias sidebar warns exactly against this; MD-2 compounds it. No golden calibration set exists to check whether the judge correlates with human judgment.
8. **Online evaluation** (Ch. 6 pp. 184–188): everything is offline/batch. No async sampled judging (5–10% of traffic), no shadow/A-B harness, no promotion flywheel wiring thumbs-down interactions back into the Langfuse dataset (the `add-from-trace` command exists — nothing automates it).
9. **Query expansion beyond synonym rewrite** (Ch. 7 territory, cheap wins first): no multi-query, no HyDE. Optional, flag-gated — but worth having behind `enabled=False` defaults given your SP12 seam already exists.
10. **Embedding-model governance** (Ch. 2 p. 46, Ch. 5): switching embedders means re-encoding everything and dimension mismatches surface as cryptic store errors (MD-10). The book recommends recording embedding model + dimension as versioned facts of the index; your manifests are the natural home and don't carry it.

---

*Fix ordering, effort estimates, and concrete designs for each item are in `03_suggestions.md`.*
