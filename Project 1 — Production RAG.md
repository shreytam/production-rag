# Project 1 — Production RAG with Eval Harness + Guardrails

> The foundation of the trio. Build a RAG system the way production teams actually build one — hybrid retrieval, reranking, an eval suite that gates CI, injection defenses, multi-tenant isolation, and a cost/latency dashboard. Projects 2 (Graph RAG) and 3 (Multimodal RAG) **reuse this project's eval harness and corpus**, which is what makes the trio a coherent portfolio story.

## Résumé bullets this produces
- "Built a production RAG system (hybrid retrieval + reranking + contextual retrieval) reaching **0.9X faithfulness / 0.8X answer-relevancy** on a held-out eval set, with a CI gate that blocks quality regressions."
- "Hardened retrieval against indirect prompt injection and enforced per-tenant ACL isolation; measured guardrail precision/recall and kept p95 latency under **X ms** at **$Y / 1k queries**."

## Concepts exercised (your vault)
RAG (whole pipeline), Evals & Observability, Guardrails & Safety, Context Engineering, Structured Outputs.

## Dataset (pick one; all are free)
- **Recommended:** a real, messy corpus you can speak to — e.g. a company's public docs, a set of arXiv papers in one area, or a government/legal corpus. ~2k–20k chunks is plenty.
- Alternatives with ready-made Q/A for eval: **HotpotQA** (multi-hop), **FinanceBench** (finance, hard), **TechQA / DocVQA-text**, or the **Natural Questions** subset.
- Make ~5% of the corpus tenant-A-only and ~5% tenant-B-only so multi-tenant isolation is testable.

## Architecture
```
ingest → chunk (+contextual prefix) → embed → vector store (+BM25)
                                                  │
query → rewrite/expand → hybrid retrieve → RRF fuse → rerank (cross-encoder)
       → ACL filter (server-side) → context assembly (budgeted)
       → generate (grounded, cited) → output guardrails → response
                                                  │
            traces + online eval + cost metrics → dashboard
```

## Recommended stack (swappable)
- **Orchestration:** plain Python first (don't hide the pipeline behind a framework); optionally LangChain/LlamaIndex later.
- **Embeddings:** an open model (BGE-M3 / Qwen3-Embedding) for cost, or a hosted one (OpenAI text-embedding-3-large / Cohere Embed v4) for quality. Support both behind an interface.
- **Vector store:** pgvector or Qdrant (both do metadata filtering for ACLs).
- **Sparse:** BM25 via the store or `rank_bm25`; fuse with **RRF (k=60)**.
- **Reranker:** a cross-encoder (BGE-reranker) or Cohere Rerank.
- **Generation:** any frontier API behind an interface; add a self-hosted vLLM path as a stretch.
- **Eval:** RAGAS + a custom LLM-judge; **promptfoo** or your own harness for the CI gate.
- **Observability:** Langfuse or OpenTelemetry GenAI traces.

## Build phases (each ends with a number to report)
1. **Baseline (naive RAG).** Fixed chunking → dense retrieval → generate. Measure faithfulness, answer-relevancy, context precision/recall on a 50–100 item eval set. *This is the number everything must beat.*
2. **Retrieval quality.** Add BM25 + RRF fusion, then a cross-encoder reranker. Report Recall@k and nDCG before/after; show the lift.
3. **Contextual retrieval.** Prepend an LLM-generated, doc-aware blurb to each chunk before indexing (Anthropic's technique). Report the retrieval-failure-rate reduction.
4. **Grounding & citations.** Enforce cited, grounded generation with structured output; add a groundedness/faithfulness check. Localize failures to retrieval vs generation.
5. **Guardrails.** Input: injection detection + PII handling. Output: groundedness gate + schema validation + citation enforcement. Report guardrail precision/recall and the latency they add.
6. **Multi-tenancy & security.** Server-side mandatory ACL filter (never trust the prompt); a test that proves tenant A can't retrieve tenant B's docs; an indirect-prompt-injection test (a poisoned doc) that your defenses catch.
7. **Ops.** Cost-per-query breakdown, p50/p95 latency budget per stage, caching (prompt + semantic), and a dashboard. Add a CI eval gate that fails the build on regression.

## Eval plan (the differentiator)
- **Retrieval:** Recall@k, MRR, nDCG against labeled relevant chunks.
- **Generation:** faithfulness, answer-relevancy, context precision/recall (RAGAS) + an LLM-judge for holistic quality, calibrated against ~20 human labels.
- **Rigor:** report confidence intervals / paired comparisons, not single point estimates; pin model + prompt versions.
- **CI gate:** a GitHub Action runs a fast eval subset on every PR and **fails** if any metric drops below baseline minus a tolerance band.

## Guardrails / enterprise checklist
ACL-aware retrieval (server-enforced) · indirect-prompt-injection defense (treat retrieved text as data, spotlighting) · PII redaction + audit logging with redaction · per-tenant rate limits · groundedness/citation enforcement on output · retention/erasure path.

## Deliverables
- A clean repo: `ingest/`, `retrieval/`, `generation/`, `guardrails/`, `eval/`, `app/`, `infra/`.
- A **README with a metrics table** (baseline → each phase, with the lift), an architecture diagram, and a "what I'd do at 100× scale" section.
- A short writeup/blog with the before/after numbers.
- A live demo (Streamlit/FastAPI) — optional but strong.

## Stretch goals
Self-hosted vLLM generation path with a cost comparison · semantic cache hit-rate tuning · an online-eval sampler on live traffic · Adaptive/Self-RAG control loop (sets up Project 2/3).

## Vault notes to reference
RAG hub; Ingestion & Chunking; Embeddings + Embedding Models and Benchmarks; Vector Stores & Indexing; Retrieval; Reranking; Query Transformation; Context Assembly & Prompting; Generation & Grounding; Advanced RAG Patterns; Evaluation + Key Evaluation Metrics + LLM-as-Judge; Statistical Significance and Eval Rigor; Failure Modes; Security; Caching; Latency & Cost Optimization; Observability & Monitoring; OWASP LLM Top 10; Prompt Injection; Input/Output Guardrails.

---

## 🚀 Build prompt (paste into Claude Code / Cursor / your coding agent)

> You are helping me build a **production-grade RAG system** as a portfolio project. Today's stack should reflect mid-2026 best practices.
>
> **Goal:** A RAG pipeline over [DATASET] that is measurably better than a naive baseline, hardened for production, and proven by an eval harness wired into CI.
>
> **Requirements:**
> 1. Implement everything behind clean interfaces (Embedder, Retriever, Reranker, Generator, Guardrail) so components are swappable. No framework lock-in for the core.
> 2. Pipeline: ingestion + chunking with an LLM-generated contextual prefix per chunk; hybrid retrieval (dense + BM25) fused with Reciprocal Rank Fusion (k=60); cross-encoder reranking; **server-side ACL filtering** (a `tenant_id`/ACL field on every chunk, enforced before similarity — never trust the prompt); token-budgeted context assembly; grounded generation with inline citations via structured output.
> 3. Guardrails: input injection detection + PII handling; output groundedness check + citation enforcement + schema validation. Treat all retrieved document text as untrusted data.
> 4. **Eval harness** (`eval/`): retrieval metrics (Recall@k, MRR, nDCG) + generation metrics (RAGAS faithfulness, answer-relevancy, context precision/recall) + an LLM-judge. Report confidence intervals, not point estimates. Include a script that compares two pipeline versions and prints a metrics table.
> 5. **CI gate:** a GitHub Actions workflow that runs a fast eval subset on each PR and fails if any metric drops below a baseline tolerance.
> 6. Observability: trace every query (retrieval hits, scores, tokens, latency, cost) to Langfuse or OpenTelemetry; expose a simple metrics dashboard.
> 7. Tests: a multi-tenant isolation test (tenant A cannot retrieve tenant B docs) and an indirect-prompt-injection test (a poisoned doc must not hijack the answer).
>
> **Deliverables:** the repo structure above, a README with a metrics table (baseline vs each improvement) and architecture diagram, and a `make eval` / `make demo` workflow.
>
> Start by proposing the repo layout and the interfaces, then implement the naive baseline + eval harness FIRST so we have a number to beat. Ask me for [DATASET] and my model/provider preferences before coding.
