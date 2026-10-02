# Book Review — *Hands-On RAG for Production* × This Codebase

**Book:** *Hands-On RAG for Production: Design, Develop, and Deploy Production-Ready RAG Applications* — Ofer Mendelevitch & Forrest Sheng Bao (O'Reilly)
**Codebase:** `Production RAG` @ branch `ui-test-console` (`3af99ff`), audited 2026-08-24
**Method:** full text extraction of the book (359 pp., per-chapter engineering digests below) + four parallel read-only code audits + first-hand verification of every headline claim. No source files were modified.

## The three deliverables

| File | What it answers |
|---|---|
| [`01_whats_right.md`](01_whats_right.md) | Where the codebase already matches or exceeds the book's playbook — verified strength-by-strength |
| [`02_whats_wrong.md`](02_whats_wrong.md) | Defects & risks (5 critical, 9 high, 16 medium + minor), plus 10 prescribed-but-missing practices |
| [`03_suggestions.md`](03_suggestions.md) | Sequenced fixes with concrete designs: Phase 0 (today), 1 (correctness), 2 (production hardening), 3 (book-driven upgrades) |

Detailed evidence (file\:line citations for every defect): `docs/review/audit_core_pipeline.md`, `audit_guardrails_security.md`, `audit_eval.md`, `audit_ops_deploy.md`.

## If you read nothing else

1. 🔴 **Rotate the NVIDIA key** leaked via `infra/.env.bak-*` (unignored, undockerignored), then fix ignores + add gitleaks.
2. 🔴 **CI never runs**: `eval-gate.yml:116` uses `secrets.*` in a job-level `if:` → whole workflow fails validation → remove that clause, run `actionlint`.
3. 🔴 **Semantic cache ignores ACL tags** → within-tenant cross-user disclosure once tagged documents exist. Fix cache identity + ACL propagation on re-ingest together.
4. 🟠 **Silent degradation paths**: reranker outage → empty context answered confidently; CLI ingest → BM25 leg silently empty (legacy pickle path); API process caches tenant BM25 indexes forever (stale reads after ingest).
5. 🟢 **Everything else is genuinely strong**: hybrid+RRF+rerank exactly per book, ACL enforced pre-similarity at every store, idempotent incremental ingest, typed PII masking, fail-closed guardrails, paired-bootstrap eval gate, JWT-only identity. The architecture matches the book's production playbook technique-for-technique — what remains is operational hardening, not redesign.

## Chapter digests (raw material)

| Digest | Book pages | Topic |
|---|---|---|
| [ch1_intro](digest_ch1_intro.md) | 2–22 | RAG fundamentals, production checklist, LangChain ref |
| [ch2_base_stack](digest_ch2_base_stack.md) | 24–62 | Parsing, chunking, embeddings, vector DBs |
| [ch3_scaling](digest_ch3_scaling.md) | 63–103 | Ingestion at scale, hybrid+rerank, guardrails, hallucination control, UX |
| [ch4_deploy](digest_ch4_deploy.md) | 105–133 | Latency budgets, security layers, caching, POC→prod |
| [ch5_platform](digest_ch5_platform.md) | 133–156 | DIY vs platform, TCO, governance |
| [ch6_eval](digest_ch6_eval.md) | 157–189 | Failure taxonomy, judges, metrics, CI gates, online eval |

Full extracted text: `book_full.txt` + per-chapter `.txt` files (page markers `<<<PAGE N>>>` preserved).
Chapters 7–10 (agents, multimodal, knowledge-enhanced, future) were extracted but not digested — out of scope for this production-focused review; the raw text is ready if you want them.
