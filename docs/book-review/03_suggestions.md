# Artifact 3 — Suggestions & Roadmap

> Companion to `01_whats_right.md` (strengths) and `02_whats_wrong.md` (defects). This file is the *what to do about it*: sequenced fixes with concrete designs, then the bigger book-inspired upgrades.
> Effort keys: **S** ≤ half day · **M** = 1–2 days · **L** = 3+ days.

---

## Phase 0 — Do today (unblocks everything, ~2–4 h total)

| # | Action | Effort | Why first |
|---|---|---|---|
| 0.1 | **Rotate the NVIDIA key** in `infra/.env.bak-1785181911`; delete the backup file; scrub the key printed inside `docs/PRODUCTION_READINESS_AUDIT.md` (delete or supersede that doc) | S | Treat as compromised from the moment it hit disk unignored |
| 0.2 | Add `infra/.env*`, `*.bak*`, `**/*.env.*` to `.gitignore` **and** `.dockerignore`; install `gitleaks` in pre-commit + a CI step | S | Prevents recurrence |
| 0.3 | **Fix `eval-gate.yml:116`**: remove `secrets.NVIDIA_API_KEY != ''` from the job-level `if:` (keep `github.repository == …`). Gate secret presence via workflow-level `env: HAS_NVIDIA_KEY: ${{ secrets.NVIDIA_API_KEY != '' }}` tested inside steps, or a tiny `check-secrets` job emitting an output consumed by `needs:` | S | Until this lands you have no CI at all — every other fix ships unprotected |
| 0.4 | Run [`actionlint`](https://github.com/rhysd/actionlint) on all workflows locally, and add it as a CI step | S | Catches this whole bug class permanently |

After 0.3 lands: run the offline lint/test job green once **before** anything else — that's your safety net for Phases 1–2.

---

## Phase 1 — Correctness & security fixes (this week–next)

Work these as small PRs in roughly this order. Items 1+2 must land together (fixing one exposes the other).

### 1.1 Make the cache ACL-aware *(M)*
Cache identity currently omits authorization scope. Minimal-diff design:

```python
# cache/semantic_cache.py
def acl_digest(acl: ACLContext) -> str:
    tags = ",".join(sorted(acl.acl_tags))
    return hashlib.blake2b(tags.encode(), digest_size=8).hexdigest() if tags else "open"

# extend lookup/store signatures:
def lookup(self, *, tenant_id, collection_id, embedding, acl_digest: str = "open"): ...
```
Thread `acl_digest(principal.to_acl())` through both tiers' Redis TAG filters (the backend already TAG-scopes on tenant/collection — add one field). Regression test to add: ingest tagged doc → query as tagged principal (warms cache) → query as tagless principal same tenant → expect **miss**, not A's answer.

### 1.2 Propagate full ACL payload on meta-only re-ingest *(S)*
In `ingest/incremental.py`, when `_meta_hash` differs, write `{"title": ..., "acl_tags": [...], "acl_open": not tags, "collection_id": ...}` instead of title only; treat a tenant change as delete-old-tenant + create-new. Test: tighten tags → re-ingest → tagless principal must lose access (dense + BM25).

### 1.3 Reranker fail-soft fallback *(S)*
Highest value-per-line in the repo:

```python
# retrieval/hybrid.py
try:
    reranked = self.reranker.rerank(query.text, window, query.rerank_top_n)
    if not reranked and window:                      # malformed 200-response guard
        raise RuntimeError("reranker returned empty ranking for non-empty input")
except Exception:
    logger.warning("reranker failed; falling back to fused order", exc_info=True)
    reranked = window[: query.rerank_top_n]
    degraded = True                                   # surface in Answer.metadata
```
Add a fake flaky-reranker test asserting fused-order fallback + `degraded=True`.

### 1.4 Cross-process sparse freshness *(M)*
Cheapest correct fix: before using a memoized tenant index, `stat()` the snapshot file and reload if mtime/size changed; unique tmp names (`f"{hash}.{os.getpid()}.{uuid4().hex}.tmp"`) for writes. Pub/sub invalidation is nicer but don't build it until mtime proves insufficient.

### 1.5 Retire the legacy pickle path *(M)*
Route `ingest/run.py` through `IncrementalIngestor` + `TenantSparseStore` (same code path as the worker); delete `PickleSparseIndexLoader` and corpus-pickle writing. This closes HG-4 (silent empty BM25 leg), HG-5 (pickle RCE surface), and removes the divergent second persistence system in one PR. If any snapshot format survives, make it JSONL of Chunk models — never `pickle.loads` on files.

### 1.6 Ingest retry policy *(S)*
Configure arq: `job_timeout=600`, `max_tries=3`, and re-raise transient-class errors (HTTP 429/5xx, timeouts) after marking FAILED so arq retries with backoff; permanent validation errors stay terminal. Add a startup sweeper that fails rows stuck in `processing/deleting` > 1 h, and compensate the blob when registry create fails.

### 1.7 Wire or delete the dead knobs *(S)*
Either thread `settings.chunk_overlap` / `max_chunks_per_corpus` into their call sites, or delete them. Delete `hybrid_require_sparse`. Decide once: config.py's docstring promises "every swappable knob" — keep the promise or soften it.

### 1.8 Small correctness batch *(S each)*
- Assert `len(vec) == embed_dimension` in the embedder; log when NIM truncation likely fired (chunk token count > model window).
- Validate `ensure_collection` against configured dimension on boot; fail fast on mismatch with a human-readable error ("collection was built for model X/dim Y").
- JWKS fetch cooldown per unknown `kid` (e.g., min 60 s between fetches, cap per minute).
- Groundedness: dedicated bounded executor + counter; log-warn when soft-fail rate > threshold in a window.
- Qdrant client: accept `api_key`, set explicit timeout, narrow the `create_payload_index` except to "already exists", pass `wait=True` on upserts where read-after-write matters.

---

## Phase 2 — Production hardening (before any real traffic)

### 2.1 Serve-ability & resilience of the API *(M)*
1. Compose: `env_file: [.env]` on `api`; document required vars next to the service; smoke-test `make up` → mint token → upload → query end-to-end.
2. `/readyz`: ping Qdrant (`count`), Postgres, Redis; report tracer status; wire into compose `healthcheck:` + Dockerfile `HEALTHCHECK`.
3. Build the pipeline in a FastAPI lifespan startup hook (kills the singleton race, makes readiness honest).
4. Exception handler middleware: map provider auth errors → 502, timeouts → 504, validation → 400; attach a request ID (also put it in logs + traces).
5. **Rate limiting + deadline:** token-bucket middleware keyed on principal + client IP; bounded semaphore (e.g., 8 concurrent pipeline calls); drop the effective request ceiling to ~60 s for interactive use (keep 600 s only for ingest-side calls). This is the book's denial-of-wallet defense — treat as mandatory, not optional.
6. Production serving story: gunicorn `-w 2-4 -k uvicorn.workers.UvicornWorker` in the image CMD (or document uvicorn `--workers`); add a `make worker` target so README quickstart matches reality.

### 2.2 Observability payoff *(S/M)*
- Open the generation stage as `as_type="generation"` with `model=ans.model`, `usage_details={input, output}`, and cost via `cost_details=` — suddenly Langfuse dashboards show real tokens/$.
- Price against `ans.model` (server-returned id), fixing the Anthropic-$0 bug.
- Warn-once logging when tracer init or span creation fails; shutdown-flush hook in lifespan; also flush at eval CLI exit.
- Record retrieved chunk ids + scores on the retrieval span (you already have them) — makes "why did this retrieve wrong?" answerable in the UI.

### 2.3 Streaming responses *(M — the single biggest UX win)*
The book treats streamed output + progress explanation as table stakes (Ch. 3 §6, Ch. 4 latency section). You already have clean seams for it:
- Extend the `Generator` protocol with `complete_stream(messages, response_model=None) -> Iterator[str]` (OpenAI SDK: `stream=True`; Anthropic: context manager events).
- `GroundedGenerator.generate_stream()` yields text deltas, then a final structured event carrying citations/usage (parse markers after the stream completes).
- New `/query/stream` endpoint (SSE via `StreamingResponse`, `media_type="text/event-stream"`): events `retrieval_done` (n hits, ms) → `token`s → `done` (citations, usage). Keep plain `/query` for the eval harness.
- Console.html already uses `textContent` discipline — append deltas to a text node; render citations from the `done` event.

### 2.4 Docs truth-pass *(S)*
Reconcile PROJECT_STATUS §2 vs §8 (trace redaction), README status vs Dockerfile reality, remove `demo` from `.PHONY`, delete/supersede the stale audit doc, note the `.claude/worktrees/` duplicate in `.gitignore` or remove it (it pollutes greps and future agents).

---

## Phase 3 — Book-driven upgrades (the roadmap)

Ordered by leverage-per-effort, per Ch. 6's own advice (satisfaction KPI first, retrieval tuning second, generator last):

### 3.1 Feedback loop + satisfaction KPI *(M)* — **start here**
The book's primary production KPI is `thumbs_up / (thumbs_up + thumbs_down)` (Ch. 6 pp. 183–184).
- `POST /feedback {interaction_id, verdict, comment?}`; persist interaction records server-side: prompt, rewritten query, retrieved chunk ids, answer, citations, feedback, timestamps, tenant.
- Log `interaction_id` on the trace so Langfuse joins naturally.
- Weekly job: satisfaction rate overall/by-topic (cheap clustering on questions), correlation of 👎 against faithfulness/relevance scores, and top-N most-downed interactions queued for review.
- Automate the flywheel you already stubbed: pipe thumbs-down interactions into `dataset_cli add-from-trace` for human labeling → golden set grows from real failures (Ch. 6's "evaluation flywheel").

### 3.2 Retrieval-quality workbench before any model swaps *(M)*
You cannot tune what doesn't measure the serving path: after Phase 1.5, produce the **live baseline run** (PROJECT_STATUS's own #1 gap), then:
- Fix eval metric hygiene first (MD-1/MD-2/MD-4/MD-17): NaN-on-failure sentinels like your Ragas adapter, judge context_precision against ground truth not self-answers, chunk-id goldens for multi-chunk corpora, refusal-aware scoring (an appropriate "I can't answer" must not read as unfaithful), special-case unlabeled trace-promoted items out of gated sets.
- Broaden what's gated: wire financebench as a second gate dataset (the adapter exists), keep the fast subset for CI speed, run full sets nightly.
- Record provenance on every run: git sha, subset ids, model versions, judge version (restores D2/MD-3) — the SP5 spec already planned this; finish it.
- Control multiplicity: gate the 2–3 critical metrics (faithfulness, recall@k, nDCG) rather than nine, or apply a Holm/Bonferroni correction.
- Then, and only then, tune chunk size / k / fusion weights against the gate — the book's "optimize the retriever before the generator."

### 3.3 Judge independence + calibration *(M)*
Point `judge_*` at a different family than `gen_*` (even just a different NIM model breaks self-preference symmetry), constrain `judge_votes` odd, keep all vote rationales as span children. Calibrate quarterly against a ~30-item human-labeled golden set; report agreement alongside every gate run (book's sidebar on judge bias, Ch. 6 pp. 164–167).

### 3.4 Diversity + structure in retrieval *(M)*
- **MMR stage** after reranking (λ≈0.7) to spend your 4000-token budget on distinct evidence instead of near-duplicates — book Ch. 3 p. 79; slot it as another `Reranker` in the chain so it stays behind the existing Protocol.
- **Small-to-big**: index 256-token chunks, but store `parent_section_id` and expand top hits to parent sections for generation. Your deterministic chunk ids + manifests make parent maps trivial. Expect better answers on multi-hop questions without sacrificing chunk-level retrieval precision.
- Expose optional **metadata filters** on `/query` (date range, source, custom fields) translated into the same pre-similarity Qdrant filter — book Ch. 2 Table 2-1. Validate field allowlists per collection.

### 3.5 Parsing pipeline upgrade *(L — do when formats demand it)*
Book Ch. 2 pp. 28–37 + Ch. 3 §2.3 prescribe a triage router: detect native-text PDF vs scanned (OCR path) vs table-heavy (`page.find_tables()` extracted as markdown blocks) vs HTML boilerplate-stripped vs VLM fallback for complex layouts (GPT-4o-class vision on high-value docs only — hallucination + cost caveats per the book). Add encoding normalization + per-parser timeouts/page caps (also fixes MD-11 adjacent worker pinning). Track per-stage parse failures as metrics — ingestion errors are invisible until they become retrieval mysteries.

### 3.6 Staging verification for re-ingestion *(M)*
Book Ch. 4 p. ~108's promote-after-check pattern: ingest into `{collection}__staging` → run automated checks (doc retrievable, chunk count sane, citation markers resolve, PII scan clean, spot-check retrieval unit tests) → atomic alias/promote → invalidate caches. Your manifest system gives you the diff; this adds the quality gate between diff and traffic.

### 3.7 Online evaluation harness *(M/L)*
Async sampled judging per Ch. 6 pp. 184–188: a background worker evaluates ~5–10% of production interactions (UMBRELA-style reference-free relevance + faithfulness) with zero user-facing latency; dashboard tracks drift between offline gate numbers and online sampled numbers. Later: shadow deployments (run challenger pipeline on live traffic sample, compare via paired bootstrap you already built) — your `eval/gate.py` becomes the A/B analyzer unchanged.

### 3.8 Embedding-model governance *(S)*
Stamp `embed_model`, `embed_dimension`, and tokenizer name into every DocManifest + collection metadata; refuse queries whose configured embedder mismatches the index (MD-10's silent-garbage failure mode becomes a loud, typed error). The book flags BYO-embedding switching cost explicitly (Ch. 5) — make the switch detectable before it's destructive.

### 3.9 Deliberately deferred (agree with the book's ordering)
Agentic RAG, multimodal ingestion, GraphRAG (Ch. 7–9): none of it pays off until the feedback loop (3.1) and live baseline (3.2) exist — the book itself gates those investments on measurement maturity, and GraphRAG's cost multiplier (Ch. 9) deserves its own eval baseline first. Also skip adopting LangChain/LlamaIndex for the core: your Protocol layer already delivers what frameworks would, without lock-in (book Ch. 5's DIY-vs-platform trade-off, resolved in favor of control).

---

## Sequenced summary

| Phase | Theme | Items | Total effort | Payoff |
|---|---|---|---|---|
| 0 | Unblock | Key rotation, ignores+gitleaks, CI fix, actionlint | ~½ day | Secrets safe, CI actually runs |
| 1 | Correctness | Cache ACL, ACL propagation, rerank fallback, sparse freshness, retire pickles, retries, knobs | ~1 week | No silent wrongness; latent disclosure closed |
| 2 | Production-grade | env_file, readiness, rate limit+deadline, streaming, observability payoff, error mapping | ~1–2 weeks | Survivable under load; debuggable in production |
| 3 | Capability | Feedback KPI, eval hygiene+baseline, judge calibration, MMR/small-to-big/filters, parsing, staging promotion, online eval, embed governance | ongoing | The book's remaining playbook |

**Definition of "production-grade" for this repo** (all verifiable):
1. CI green including eval gate vs a recorded, provenance-stamped baseline.
2. No credential reachable by `git add -A` or `docker build` (gitleaks clean).
3. Cache + stores enforce identical ACL semantics (tagged-doc regression test passes both paths).
4. Any single component outage degrades (flagged) instead of silently mis-answers: rerank→fused order, groundedness→soft-fail alert, tracing→warn-once.
5. `make up` serves an authenticated query end-to-end in containers, with readiness probes and rate limits active.
6. Every production interaction stored with feedback hooks; satisfaction rate visible on a dashboard next to faithfulness and cost.

That last line is Ch. 6 in one sentence: *metrics aligned to use-case goals, layered — automated + human, continuously.*

---

*Supporting detail: `docs/review/audit_core_pipeline.md`, `audit_guardrails_security.md`, `audit_eval.md`, `audit_ops_deploy.md` carry file\:line evidence for every defect referenced here.*
