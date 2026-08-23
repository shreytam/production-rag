# Guardrails & Security Audit — Production RAG

**Area:** Guardrails, authentication/authorization, multi-tenancy, PII handling, semantic-cache tenancy
**Mode:** Read-only static audit (no code executed against live services)
**Date:** 2026-08-24 · **Auditor:** ox-alpha (automated subagent audit)
**Codebase:** `/Users/shreytam/Ai/ai_projects/Production RAG` @ `3af99ff`

---

## 1. Scope & Method

Files examined (primary):

| Layer | Files |
|---|---|
| Guardrails | `guardrails/runner.py`, `citation_enforcement.py`, `input_injection.py`, `output_groundedness.py`, `pii_guard.py`, `schema_validation.py` |
| PII providers | `providers/pii/regex_detector.py`, `presidio_detector.py`; backing logic in `ingest/pii.py`, `ingest/audit.py`, `ingest/run.py` (`_apply_pii_ingest_policy`) |
| Auth | `app/auth.py`, `app/api.py`, `app/ui.py`, `providers/auth/jwt_verifier.py`, `dev_signer.py`, `allowlist.py`, `scripts/mint_token.py`, `core/config.py` validators |
| Tenancy | `retrieval/acl.py`, `core/types.py` (`ACLContext`/`Principal`), `providers/vectorstores/qdrant_store.py`, `providers/sparse/bm25.py`, `tenant_store.py` |
| Documents/storage | `app/documents.py`, `providers/docstore/postgres.py` (+`memory.py`), `providers/blobstore/local_disk.py`, `providers/manifest/jsonl_store.py` |
| Async worker / cache | `ingest/worker.py` (the brief's "cache/worker.py" resolves here — there is no `cache/worker.py`), `cache/semantic_cache.py`, `cache/_redisvl_backend.py` |
| Supporting | `core/pipeline.py`, `generation/grounded_generator.py`, `generation/prompts.py`, `eval/generation_metrics.py` (`faithfulness`), `observability/langfuse_tracing.py`, `retrieval/hybrid.py` |

Method: full read of every listed file, trace of the query path (`POST /query` → auth → guardrails → cache → retrieval → generation → output guards → scrub → cache-store) and the ingest/delete path (`POST /documents` → blob → arq worker → stores), cross-checked against the security test suite (`tests/test_auth.py`, `test_multitenant_isolation.py`, `test_prompt_injection.py`, `test_guardrails.py`, `test_pii_*.py`, `test_blobstore.py`, `test_stores_acl.py`, …) and README security claims. All line references below are to the current working tree.

## 2. Executive Summary

**Verdict:** This is an unusually security-conscious RAG codebase. Identity is derived exclusively from pinned-algorithm verified JWTs, ACL filtering is pushed server-side *before* similarity scoring, path-traversal surfaces are hashed away, and the guardrail runner fails closed with thorough output scrubbing. A dedicated test suite covers algorithm-confusion, `alg:none`, cross-tenant isolation, poisoned-document hijack, and PII overlap handling.

That said, the audit found **3 high-severity issues**, **~9 medium**, and several low/latent defects. The most consequential:

1. **[H-1] The semantic cache is not ACL-tag-scoped.** Cache entries are keyed only by `(tenant_id, collection_id, embedding)`. Two principals in the same tenant with *different* `acl_tags` can be served each other's answers — and, via the retrieval tier, raw chunk text their tags do not permit. This breaks the tag half of the tenancy model that retrieval enforces rigorously.
2. **[H-2] JWKS cache refreshes on every unknown `kid` with no cooldown** — an unauthenticated attacker can force a JWKS fetch per request (DoS amplification against the verifier and the IdP).
3. **[H-3] Groundedness fail-open degrades into permanent silent bypass under load**: the shared 4-worker pool starves when judge calls are slow, after which every answer passes unverified.

Finding index: **H-1, H-2, H-3 · M-1 … M-9 · L-1 … L-7**

---

## 3. How the Security Surface Fits Together

Query path (`app/api.py:79`, `core/pipeline.py:100`):

```
POST /query
 └─ require_principal        app/auth.py:35   Bearer → JWTVerifier.verify (alg pinned) → allowlist intersect → frozen Principal
 └─ pipeline.answer          core/pipeline.py:100
     ├─ input guards         InjectionGuardrail (tiered regex, optional LLM escalation)
     │                       PIIGuardrail (REDACT payload applied before anything else sees the text)
     ├─ tracer span opened   only AFTER redaction (raw question never enters tracing)
     ├─ query rewrite        retrieval-only; original question kept for generation
     ├─ answer-tier cache    lookup(tenant_id, collection_id, embedding)   ← gap H-1 lives here
     ├─ retrieval            Qdrant pre-similarity ACL filter + per-tenant BM25 + acl_predicate
     │   └─ indirect-injection scan of chunk texts (detect-only, M-1)
     ├─ generation           spotlighted <context> delimiting, [n] markers resolved to chunk_ids
     ├─ output guards        CitationGuardrail → SchemaGuardrail → PIIGuardrail? → GroundednessGuardrail
     └─ block scrub          refused=True + generic message + metadata/reason stripping (pipeline.py:274–286)
```

Ingest path (`app/documents.py:118`, `ingest/worker.py:63`): validated upload → sha256-namespaced blob → registry row → arq job → parse → PII policy (fail-closed at ingest) → chunk → tenant-scoped stores → cache eviction (own + unscoped partition).

## 4. Strengths (with evidence)

- **S-1 · Single, verified identity path.** `Principal` is built only from a verified token's claims and is a frozen Pydantic model (`core/types.py:50–70`); `QueryRequest` deliberately carries no identity field (`app/api.py:59–63`); `require_principal` rejects non-Bearer headers with 401 + `WWW-Authenticate` (`app/auth.py:40–48`). No header/body identity path exists anywhere.
- **S-2 · Algorithm pinned from config, never from the token.** `jwt.decode(..., algorithms=[self._alg])` with `require: ["exp"]` (`providers/auth/jwt_verifier.py:100–110`). The test suite proves both `alg:none` and an HS-forgery-using-the-RSA-public-key are rejected (`tests/test_auth.py:130`, `:242`) — textbook algorithm-confusion coverage.
- **S-3 · Prod configuration fails fast.** `_validate_auth` refuses prod boots without HS256 secret / RS256 JWKS / issuer+audience and forbids `auth_dev_signer_enabled` in prod (`core/config.py:231–243`); `_validate_pii_settings` requires a salt when audit-value hashing is on (`core/config.py:245–251`).
- **S-4 · Fail-closed defaults everywhere it matters.** Guard exceptions → BLOCK unless a guard opts out (`guardrails/runner.py:112–119`); malformed LLM injection verdicts → BLOCK (`input_injection.py:79–81`); unknown JWKS kid or fetch failure → 401 (`jwt_verifier.py:49–96`); unknown tenant under `StaticAllowlist` → all tags dropped (`allowlist.py:24–26`); unsupported upload content-type rejected before reading the body (`app/documents.py:133–136`); ingest errors mark documents FAILED, never half-ready (`ingest/worker.py:96–99`, `:117–120`).
- **S-5 · ACL enforced pre-similarity, server-side.** Qdrant search passes `query_filter=qdrant_filter(acl)` so tenant/tag/collection filtering happens *before* vector scoring — no post-filter leakage (`providers/vectorstores/qdrant_store.py:132–139`, `retrieval/acl.py:20–67`). Deletes and metadata updates are equally scoped via `HasIdCondition AND qdrant_filter` (`qdrant_store.py:153–173`). BM25 isolation is structural: only `acl.tenant_id`'s index is consulted, then `acl_predicate` for tags (`providers/sparse/bm25.py:53–73`), matching `ACLContext.allows()` as single source of truth (`core/types.py:37–47`, `retrieval/acl.py:70–81`).
- **S-6 · Path traversal designed out.** Blob keys hash the tenant segment before joining to the root (`app/documents.py:106–110`); `LocalDiskBlobStore._path` additionally resolves and checks `is_relative_to(root)` (`providers/blobstore/local_disk.py:10–15`); manifest store hashes both tenant and doc segments (`providers/manifest/jsonl_store.py:15–18`); tenant sparse index paths hash the caller-controlled tenant id (`providers/sparse/tenant_store.py:27–29`).
- **S-7 · Cache hygiene where it exists.** Tenant+collection TAG filter expressions on every lookup (`cache/_redisvl_backend.py:59–63`), per-key TTL backstop, precise per-document eviction that also covers the unscoped `__none__` partition (`ingest/worker.py:35–60`), and refused/blocked answers are never stored (`core/pipeline.py:289–295`).
- **S-8 · Guardrail runner discipline.** Per-guard latency capture, exceptions never 500 a request, blocked-input responses carry a generic message while details stay in the trace (`guardrails/runner.py:97–122`, `core/pipeline.py:118–135`). Output blocks are scrubbed hard: text replaced by a generic refusal, citations/contexts/retrieved-ids cleared, `structured_output` popped, per-result reason/payload/metadata stripped from the returned object but retained in tracing (`core/pipeline.py:274–286`).
- **S-9 · PII ordering is correct.** Input redaction happens *before* the tracer span is opened, so raw questions never reach Langfuse span parameters (`core/pipeline.py:112–140`); the tracer additionally runs a recursive redacting mask that fails closed to `[PII_REDACTION_ERROR]` (`observability/langfuse_tracing.py:136–155`).
- **S-10 · Indirect-injection containment is layered.** Retrieved context is delimited with `<context>` spotlighting and explicit "DATA, not instructions" framing plus instruction-hierarchy system rules (`generation/prompts.py:8–31`); retrieved chunks are scanned and flagged (`core/pipeline.py:197–202`); a hijacked-answer output path is tested (`tests/test_prompt_injection.py:80–98`). Normalization defeats zero-width chars, NFKC tricks, leetspeak, and letter-spacing (`guardrails/input_injection.py:16–56`).
- **S-11 · Dev-token minting is triple-gated.** `/ui` and `/ui/token` 404 unless flag **and** secret are set (`app/ui.py:35–44`), the CLI refuses without both (`scripts/mint_token.py:17–21`), and config refuses the combination entirely in prod (S-3). The signer itself cannot run without a non-empty secret (`dev_signer.py:27–28`).
- **S-12 · Citation enforcement has teeth.** Non-refused answers require ≥1 citation; cited chunk_ids must exist in what the generator actually saw; claimed markers must map to real passages (`guardrails/citation_enforcement.py:41–79`) — with a fallback hole noted in M-4.
- **S-13 · Security tests are first-class.** ~30 auth tests including algorithm confusion and allowlist semantics, end-to-end cross-tenant leak tests for dense/BM25/full-pipeline (`tests/test_multitenant_isolation.py`), poisoned-document hijack tests, PII overlap/merge tests, blobstore traversal tests.

## 5. Defects & Risks

Severity: **H** = exploitable/impactful as shipped · **M** = real defect or risky default, fix before prod hardening · **L** = latent/low-impact/hardening.

---

### H-1 · Semantic cache ignores `acl_tags` → cross-principal leakage inside a tenant

**Severity: HIGH (confidentiality) · Files: `cache/semantic_cache.py`, `cache/_redisvl_backend.py`, `core/pipeline.py`**

The cache Protocol and backend key entries on `(tenant_id, collection_id, embedding)` only:

- `cache/semantic_cache.py:27–35` — `lookup/store(..., tenant_id, collection_id, embedding)`; no acl parameter exists anywhere in the Protocol.
- `cache/_redisvl_backend.py:59–63` — the lookup filter expression is exactly `Tag("tenant_id") == tenant_id & Tag("collection_id") == norm_collection(collection_id)`. The schema (`_redisvl_backend.py:32–44`) defines no ACL field.
- `grep -rn "acl" cache/*.py` returns nothing.

Consequence: within one tenant, a principal holding `acl_tags=("finance",)` asks a question; the generated answer (citing finance-tagged chunks) is cached. A *tag-less* principal in the same tenant asking a semantically similar question (cosine similarity ≥ `cache_similarity_threshold`, default 0.9) receives that answer verbatim via the answer tier (`core/pipeline.py:161–169`). The retrieval tier is worse: the cached payload is the serialized `ScoredChunk` list *including chunk text* (`semantic_cache.py:46–51`), so the tag-less caller gets finance-gated chunk content their own retrieval would have filtered out (`pipeline.py:181–187`). Retrieval's rigorous tag enforcement (`retrieval/acl.py`) is bypassed entirely on a cache hit.

Aggravating factors: `collection_id=None` means "all collections in the tenant" at retrieval time (`retrieval/acl.py:34–36`) and maps to the shared `__none__` cache partition, widening the overlap surface; document deletion does evict precisely (`ingest/worker.py:35–60`), but eviction fixes staleness, not authorization.

**Recommendation:** add an `acl_scope` TAG field to both index schemas — e.g., a digest of the caller's *sorted effective tag set* (after allowlist intersection), so callers with identical grants share entries and others never do; include it in lookup/store/invalidate filters. Alternatively partition per `(tenant, collection, tags-digest)`. Add a cross-tag cache test to `tests/test_cache_pipeline.py`.

---

### H-2 · JWKS cache refreshes on every unknown `kid` — unauthenticated fetch amplifier

**Severity: HIGH (availability) · File: `providers/auth/jwt_verifier.py:43–51`**

```python
def get(self, kid: str):
    if not self._keys or time.time() >= self._expires_at:
        self._refresh()
    if kid not in self._keys:
        self._refresh()  # unknown kid → single refresh (key rotation)
```

There is no cooldown, jitter, or negative caching on the unknown-kid path. Any unauthenticated client can send tokens with random `kid` values at line rate; each triggers a synchronous JWKS HTTP fetch (5 s timeout) to the IdP from inside request handling. Effects: (a) the verifier becomes an inadvertent DoS tool against the IdP; (b) request latency inflates by up to 5 s per attempt while threads block; (c) if the IdP rate-limits the fetcher, `_refresh` raises → all traffic 401s (fail closed, but now trivially remotely triggerable).

**Recommendation:** cap forced refreshes (e.g., min interval 60–300 s with jitter, remember last-refresh time even on failure); negative-cache recently-seen bogus kids; make refresh asynchronous/best-effort for unknown kids after the first miss.

---

### H-3 · Groundedness guardrail silently degrades to permanent bypass under load

**Severity: MEDIUM-HIGH (safety availability) · Files: `guardrails/output_groundedness.py:16–20, 67–82`; `eval/generation_metrics.py:111–164`**

Fail-open on timeout is a documented, deliberate tradeoff ("a slow judge must not mass-block real answers"). But the mechanism turns it into an attacker-amplifiable permanent bypass:

- The module-level pool is bounded at 4 workers (`output_groundedness.py:20`). On timeout the future is *abandoned, not cancelled* — it keeps running to completion (comment at lines 16–19 acknowledges double cost).
- Under sustained slow judge conditions (LLM provider degradation, or simply ≥4 concurrent long answers), workers stay occupied by zombie faithfulness calls; new submissions queue behind them and every `fut.result(timeout=20)` times out → PASS with `groundedness_unverified`. There is no recovery signal, metric, or alert hook distinguishing "one timeout" from "guardrail effectively disabled."
- Failure modes are also asymmetric: a judge that returns *unparseable* output makes `faithfulness` return `0.0` (claims=[] → 0.0, `generation_metrics.py:141–142,161–162`) → BLOCK, while a judge that *hangs* → PASS. Availability thus depends on how the upstream fails, which is not a property the codebase controls.

**Recommendation:** bound total outstanding submissions and shed load deterministically (e.g., semaphore → immediate PASS-with-metric when saturated); emit a counter/alarm on consecutive timeouts; consider a cheap deterministic fallback (embedding-overlap heuristic) when the judge is unavailable instead of binary pass.

---

### M-1 · Indirect prompt injection in retrieved chunks is detect-only

**Severity: MEDIUM · File: `core/pipeline.py:197–202`**

Retrieved chunk text matching strong injection patterns sets `indirect_injection_suspected` metadata and logs a warning — generation proceeds regardless. A tenant who can plant a document (the system is explicitly product-driven: tenants upload their own files) can embed `ignore previous instructions…` payloads that ride into the `<context>` block on every future query over that doc. Remaining defenses are spotlighting delimiters and system-prompt instruction hierarchy (S-10) — necessary but not sufficient against strong injection corpora; the groundedness/citation guards then check factual grounding of whatever the model produced, not whether it obeyed injected instructions.

**Recommendation (defense-in-depth):** strip or neutralize strong-pattern matches from context text before assembly (redaction preserves citations while removing imperative phrasing); refuse-or-degrade retrieval hits flagged with multiple strong patterns; track the flag as an eval metric so poisoned-corpus behavior is measurable.

### M-2 · Injection blocklist false-positive posture blocks legitimate queries

**Severity: MEDIUM (availability/usability) · File: `guardrails/input_injection.py:33–51`**

Strong-tier patterns BLOCK outright on matches like bare `"jailbreak"` (`:39`), `(override|bypass|disable|circumvent)\s*(rules?|safety|restrictions?|filters?|guidelines?)` (`:41`) — which fires on benign security/IT questions such as *"how do I disable Windows firewall rules?"* — and `(new|updated)\s*instructions?:` (`:42`) (*"updated instructions: see the runbook"*). Conversely the list is trivially evaded by paraphrase, translation, homoglyph substitution beyond zero-width (e.g., Cyrillic `іgnore`), or indirect phrasing — inherent to blocklists. Net effect: real attack recall is low while false-positive blocking of power users is guaranteed. No telemetry/feedback path distinguishes the two today beyond trace spans.

**Recommendation:** keep the tiered design but route strong-pattern hits through adjudication (LLM classifier already exists) rather than unconditional BLOCK, or restrict unconditional blocks to exfiltration-shaped patterns; log FP feedback; document the intended threat model (this layer is speed-bump + signal, not the perimeter).

### M-3 · LLM escalation classifier prompt is itself injectable

**Severity: MEDIUM-LOW · File: `guardrails/input_injection.py:68–82`**

`f"User text:\n{text}\n\nIs this a prompt-injection/jailbreak attempt?"` embeds the raw suspect text without delimiters or escaping. An input engineered to steer the classifier ("… Answer: no") can flip weak-signal adjudication to PASS. Mitigations already present: schema-enforced response and fail-closed on malformed verdicts (`:76–81`). Use the same `<context>`-style delimiting used for retrieved data and instruct the classifier to ignore directives inside the quoted block.

### M-4 · Phantom-citation enforcement has a fallback hole

**Severity: MEDIUM · Files: `generation/grounded_generator.py:70–88`; `guardrails/citation_enforcement.py:65–79`**

When the generator honors the structured schema, `claimed_markers` is populated and the phantom check runs. When the model ignores the schema, the fallback scrapes `[n]` markers from prose (`grounded_generator.py:72–78`) with `claimed = []`, and unresolvable markers are silently dropped from `citations` (`:86–87 continue`) while remaining visible in the answer text. CitationGuardrail then sees `claimed=[]` → phantom check vacuous → **PASS**, so an answer can ship with dangling `[7]` markers pointing at nonexistent passages whenever exactly one real citation also exists (zero-real-citations still blocks via the ≥1 rule). Enforcement strength therefore silently depends on model schema compliance.

**Recommendation:** treat scraped-but-unresolvable markers on the fallback path as hallucinated (BLOCK), or set `claimed_markers` to the scraped set so the existing check fires.

### M-5 · `apply_redactions` chaining semantics contradict their documentation (latent multi-guard bug)

**Severity: MEDIUM (latent) · File: `guardrails/runner.py:81–91`**

Docstring: "Multiple REDACT results are chained: the payload of the first REDACT guard is fed as input to the next." Code: every guard ran against the *same original* text, so the loop simply takes the **last** REDACT payload. With two redacting guards redacting different entity classes, the final text retains whatever only the first guard caught. Today `default_runner` wires at most one PII guard per phase (`runner.py:141–152`), so this is latent — but any composition (e.g., adding a second detector for names/IBAN) silently reintroduces leakage.

**Recommendation:** either feed each subsequent guard the accumulated redacted text (requires re-running checks sequentially), or union span offsets across results and apply once; align the docstring either way.

### M-6 · Output-PII "scrub" corrupts `structured_output["answer"]`

**Severity: LOW-MEDIUM · File: `core/pipeline.py:236–240`**

On REDACT, the pipeline re-applies `apply_redactions(raw_meta_ans, out_results)` to the structured copy — but `apply_redactions` ignores its first argument and returns the payload computed against the **main** `ans.text`. The structured field is replaced wholesale by the redacted main-answer string: correct only when the two strings are identical, corrupting the field otherwise (SchemaGuardrail validated the *original* dict earlier, so validation no longer describes the shipped object). Not a leak (replacement ⊆ redacted main text) but a correctness defect in a compliance-relevant path.

### M-7 · Least-privilege allowlist is off by default

**Severity: MEDIUM (deployment risk) · Files: `core/config.py:125`; `providers/auth/allowlist.py:10–12`; `core/registry.py:150–159`**

Default is `NullAllowlist` — token claims pass through untouched. Tag-level authorization then rests entirely on the IdP minting correct claims. The safer primitive (`StaticAllowlist`, unknown-tenant→∅) exists and is tested but requires opt-in via `acl_allowlist_source`. Given H-1 amplifies any over-broad tag claim, production deployments should be nudged: warn loudly (or refuse in `app_env=prod`) when running without an allowlist source.

### M-8 · PII detection coverage and audit durability fall short of the compliance framing

**Severity: MEDIUM · Files: `providers/pii/regex_detector.py`; `guardrails/pii_guard.py:20–29`; `core/config.py:136–143`**

- Default detector is regex with four patterns: EMAIL, US-style PHONE, SSN, CREDIT_CARD. No person names, addresses, IBAN/API keys/dates of birth; phone pattern misses most international formats; CREDIT_CARD has no Luhn check (16-digit account/order numbers starting 4/5/34/37/6 misclassify); Presidio (which covers names/addresses) is optional and off by default. `pii_mode="keep"` additionally leaves raw PII *in the index* tagged only via `metadata["pii_types"]` (`ingest/worker.py:85–91`).
- `PIIGuardrail.audit_log` is a per-instance, in-memory, **unbounded** list (`pii_guard.py:20–29`): two separate instances exist per runner (input + optional output copy), growth is proportional to findings (memory leak under load), and everything is lost on restart — so the docstring's "audit log … for compliance purposes" overstates durability. The ingest-side `PIIAuditLog` does write JSONL to disk, but only at ingest time.
- Positive notes worth keeping: findings recorded are value-free (type/start/end only, `ingest/pii.py:47–55`); salt-required validator when hashing values (S-3); overlap merging prevents un-redacted tails (`ingest/pii.py:14–27`).

**Recommendation:** enable Luhn; document pattern scope honestly; bound/persist the runtime audit trail (route guardrail findings into `PIIAuditLog`-style JSONL with rotation); reconsider `keep` mode for multi-tenant prod.

### M-9 · No rate limiting / cost throttling on LLM-backed endpoints

**Severity: MEDIUM · Files: `app/api.py:79–100`; `core/config.py:83–88, 114–115`**

A valid token unlocks up to ~4 auxiliary LLM calls per query (query rewriter, injection escalation, faithfulness extraction+verdict) plus the main generation call, with `request_timeout_seconds=600` and retries=5 configured. `/query` is a sync `def` (threadpool-executed), so slow generations consume threadpool slots; there is no per-principal/per-tenant quota, concurrency limit, or burst protection anywhere in the app. Uploads likewise lack per-tenant quotas (`max_chunks_per_corpus` applies to CLI ingest, not API uploads). Cost-amplification DoS by any authenticated principal is feasible; combined with dev-console minting on non-prod deploys, effectively unauthenticated there.

### L-1 … L-7 · Lower-severity findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| L-1 | Low | Cache key uses the embedding of the **rewritten** retrieval question while the payload answers the **original** question; near-duplicate rewrites (≥0.9 cosine) of different questions can return each other's answers. Also a literal collection id `"__none__"` from a CLI corpus would collide with the sentinel partition. | `core/pipeline.py:158–160` vs `:205`; `cache/semantic_cache.py:18–23` |
| L-2 | Low | `doc_ids` TAG joined with `\|`; any doc_id containing `\|` breaks per-document eviction targeting (API doc_ids are uuid4-hex and safe; external corpus doc_ids are unsanitized). | `_redisvl_backend.py:38, 80` |
| L-3 | Low | Manifest `save()` uses a fixed `<name>.tmp` — concurrent saves of the same manifest can interleave into a torn tmp before `os.replace`; `load()` swallows all exceptions → silent re-ingest on corruption. | `providers/manifest/jsonl_store.py:24–42` |
| L-4 | Low | Tenant sparse store deserializes **pickle** files from `.cache/sparse_tenants/` — RCE-if-write-access trust assumption; fine for single-host deploys, worth noting in threat model. | `providers/sparse/tenant_store.py`, `pickle_loader.py` |
| L-5 | Low | Postgres registry interpolates `{table}` from config into DDL/DML via f-strings (config is a trusted boundary; all *values* are parameterized). New connection per call — no pooling. Default DSN embeds dev credentials `rag:rag@localhost`. | `providers/docstore/postgres.py:5–46`; `core/config.py:46` |
| L-6 | Low | `redis_password` setting exists but is referenced nowhere — Redis runs unauthenticated unless the password is embedded in `redis_url`; anyone with queue access can enqueue ingest/delete for arbitrary document ids (worker resolves tenant from the privileged row). | `core/config.py:182` (sole reference); `ingest/worker.py:182`; `app/documents.py:89–103` |
| L-7 | Info | Input-blocked refusals embed full input-guard results (incl. matched pattern labels) in `Answer.metadata["guardrails"]` (`pipeline.py:86–98`) — stripped from API responses but visible to in-process consumers/demo surfaces. Output-block scrubbing (S-8) does not have this gap. Upload size check happens after full body read (spooled to disk, not memory) with no streaming cap; blobs stored plaintext at rest. | `app/documents.py:138–149`; `local_disk.py` |

---

## 6. Test Coverage Assessment

Strong for what it covers, with three gaps that mirror the findings:

**Covered well:** algorithm confusion & `alg:none` rejection, expired/mismatched-audience tokens, allowlist intersection semantics, prod-config fail-fast (`tests/test_auth.py`); dense/BM25/full-pipeline cross-tenant isolation incl. live Qdrant gate (`tests/test_multitenant_isolation.py`); poisoned-doc hijack end-to-end and spotlighting presence (`tests/test_prompt_injection.py`); guardrail blocking/fail-closed/fail-soft matrix (`tests/test_guardrails.py:397–438`); PII overlap/redaction/audit hashing (`test_pii_*.py`, `test_output_redact.py`); blobstore traversal (`test_blobstore.py`); cache tenant/collection scoping and eviction (`tests/cache/`, `test_cache_worker.py`).

**Gaps:** (1) no test exercises two principals with *different* tags hitting the same cache partition (H-1 would be caught by one); (2) no test for repeated unknown-`kid` JWKS behavior (H-2); (3) no test for groundedness behavior when the pool is saturated (H-3); (4) no adversarial tests of injection-evasion homoglyphs beyond zero-width or FP-rate fixtures for M-2.

---

## 7. Prioritized Remediation Roadmap

1. **Now — H-1:** add ACL-scope tag to both cache schemas + filters + invalidation; add cross-tag isolation test. (Small change, closes the only confidentiality finding.)
2. **Now — H-2:** cooldown/negative-cache on JWKS unknown-kid refresh.
3. **Next — H-3:** bounded outstanding-work shedding + consecutive-timeout metric for groundedness; unify judge-failure semantics (decide BLOCK vs PASS for unparseable, document it).
4. **Next — M-4, M-5, M-6:** close the phantom-marker fallback hole; fix redaction chaining (or its docstring); replace structured-output scrub with offset-correct targeted redaction.
5. **Hardening pass — M-1, M-3:** neutralize strong-pattern matches in context assembly; delimit the classifier prompt.
6. **Policy/config — M-2, M-7, M-8, M-9:** tune blocklist adjudication; require allowlist source in prod; persist/bound runtime PII audit trail + Luhn; add per-principal rate limits (e.g., slowapi/Redis token bucket keyed on tenant_id+sub).
7. **Hygiene — L-1…L-7:** rewrite-vs-original cache key decision, tag-separator safety, unique tmp names, pickle→safer format note, wire or delete `redis_password`, empty default DSN + validator.

---

*Audit performed read-only; no code was modified. All file/line references verified against working tree @ `3af99ff`.*

