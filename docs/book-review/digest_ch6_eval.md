# Engineering Digest — Chapter 6: Evaluating Your RAG Application

**Book:** *Hands-On RAG for Production* — Mendelevitch & Bao
**Source:** `docs/book-review/ch6_eval.txt` (1336 lines; chapter body ≈ book pp. 157–189)
**File notes:** the excerpt opens mid-sentence (first partial page missing) and its last ~24 lines are the opening of Ch. 7 "From RAG to AI Agents" (p. 191) — excluded from this digest. Code samples referenced by the book live in its companion GitHub notebooks.

## What this chapter covers
A production-oriented treatment of RAG evaluation: a failure-mode taxonomy across the three pipeline stages (retrieval → generation → ingestion), LLM-as-a-judge mechanics and pitfalls, classical IR metrics vs. reference-free metrics (UMBRELA, AutoNuggetizer, HHEM), a survey of evaluation platforms (Open RAG Eval, Ragas, DeepEval, Amazon Bedrock), human-feedback KPIs, and how to wire all of it into offline CI gates + online monitoring, including latency/uptime/cost system metrics.

## Key takeaways (TL;DR)
- **Optimize the retriever before the generator.** "If your retrieval recall is 40%, no amount of prompt engineering will make your RAG application successful." Hybrid search (vector + BM25) and rerankers are the first levers to pull.
- **Two metric families, two cost models:** classical IR metrics are cheap but need human-curated *golden chunks/answers* that go stale; reference-free LLM-as-a-judge metrics (UMBRELA, AutoNuggetizer) and hallucination detectors (HHEM) remove labeling but add inference cost/latency.
- **Distinguish faithful-but-incorrect from unfaithful incorrectness** — the former points at stale ingestion data, the latter at the generator. Troubleshooting depends on this split.
- **Run eval as a CI gate offline** (no-regression policy on faithfulness/retrieval relevance; P95 latency ≤ +5%) and **async online** on ~5–10% sampled traffic so judging never blocks the user.
- **Make thumbs-up/down satisfaction rate your primary KPI**, then correlate it against automated metrics — if low faithfulness doesn't correlate with 👎, your proxy metric is misleading you.
- **Version everything that measures**: benchmark datasets, judge prompts, and judge model versions — otherwise scores drift because the yardstick moved, not because the system did ("evaluation drift").
- Judge models inherit biases (self-referential/"sounds like AI", verbosity-as-quality, pro-AI overestimation) and are stochastic even at temperature 0 → prefer pairwise comparisons, average multiple runs, calibrate against a golden test set.

## Failure-mode taxonomy (pp. ≈158–163)
Failures cluster by pipeline stage. The book's Table 6-1, condensed:

| Stage | Failure mode | What it looks like | Mitigation |
|---|---|---|---|
| Retrieval | Failure to retrieve (low recall) | Info exists but isn't surfaced — complete miss or partial miss; "blinds" the LLM, which then either refuses or hallucinates from pre-training | Hybrid search (vector + BM25), reranker |
| Retrieval | Irrelevant retrieval (low precision) | Retriever is "noisy": pulls topically-related chunks lacking the answer facts (e.g., GitHub-general chunks for a GitHub-security question), polluting context | Hybrid search, reranker |
| Retrieval | Architectural limits | Multi-hop queries ("HQ of the company that acquired Startup X") and broad sensemaking queries defeat single-shot semantic matching | Agentic RAG (Ch. 7), knowledge graphs (Ch. 9) |
| Generation | Faithfulness failure / hallucination | Answer contradicts or exceeds the retrieved chunks (deadline July 31 stated as "early August"); most credibility-destroying error | Hallucination detection (e.g., HHEM), LLM-judge rubrics |
| Generation | Context utilization failure | Relevant chunks supplied but ignored (ordering bias / lost-in-the-middle); answer grounded yet dangerously incomplete (risks chunk dropped from Project Atlas answer) | Prompt engineering so all chunks get weighted; detect via AutoNuggetizer |
| Generation | Answer relevance failure | Faithful + complete but non-responsive ("features A, B, C" when asked "is it safe?"); ROUGE-L/BERTScore miss it — use a custom LLM-judge rubric | LLM-as-a-judge custom rubric |
| Ingestion | Structural parsing errors | Tables/charts/images in PDF/DOCX/PPTX flattened to word salad → systematically low retrieval relevance on visual docs; catch via robust ingestion logging + specialized parsers | Logging, specialized parsers per format |
| Ingestion | Content staleness | Superseded docs retrieved as current (year-old policy answered as fact) | Unique entity ID + version number per doc; purge prior-version chunks on re-ingest; proactive test: two versions of one entity ID = ingestion bug |

Engineering notes:
- Ingestion failures are insidious: they create a false sense of reliability while corrupting retrieval/generation downstream. The entity-ID+version scheme doubles as a canary test.
- The book's worked example: missing Chunk B (72-hour risk-form deadline) made the LLM say "no timeframe specified" plus an irrelevant celebration-lunch aside — a partial-recall failure that reads as a compliance trap for users, not just a wrong answer.

## LLM-as-a-Judge (pp. ≈164–167)
**Mechanics:** give a judge model the query, generated answer, and explicit criteria; get back scores/ratings/critique. Its core advantage over fixed metrics is *flexibility* — arbitrary natural-language rubrics (persona adherence, causal reasoning, "act as a skeptical expert and verify the answer is fully supported by context"). Research cited (Zheng et al., Balog et al.): top-tier judge models correlate well with human preferences **under optimal configurations** — not plug-and-play.

**Known limitations to design around:**
- **Task-dependent alignment:** expert-level on creative/summary tasks, weaker on rigid mathematical logic.
- **Pointwise vs pairwise:** rating one response is common, but asking the judge to *compare two responses* yields more stable, human-aligned judgments.
- **Self-referential bias** (dedicated sidebar): judges favor outputs that mimic LLM training-data style → style-over-substance, verbosity-as-quality ("the verbosity trap"), shared blind spots (won't penalize errors it also makes). Mitigation: calibrate against a human-verified golden test set.
- **Overestimation ("pro-AI") bias:** inflates scores of LLM-generated text vs deterministic/classic output; may require manual ground truth in the prompt.
- **Cost & latency:** frontier-model calls make judge-based metrics impractical for massive iterative test cycles.
- **Stochastic instability:** same input, different scores across runs — even temperature=0 / fixed seeds aren't foolproof. Workarounds: run N times and average; switch to structured pairwise comparisons.

**Reference implementation pattern (book code):**
```python
def evaluate_with_llm_judge(query, context, generated_answer, model="gpt-4o"):
    prompt = f"""You are an impartial judge ... two criteria:
    1. Factuality: grounded only in provided context, no contradictions
    2. Answer Relevance: relevant and helpful for the query
    Score each 1–5 with brief explanations. Respond ONLY with JSON:
    {{"factuality_score", "factuality_reasoning",
      "relevance_score", "relevance_reasoning"}}"""
```
Engineering details worth copying:
- System message pins the model to valid-JSON-only output; `temperature=0`.
- Response parsed defensively: `re.search(r'\{.*\}', text, re.DOTALL)` strips markdown fences before `json.loads`; return `None` + log on parse/API failure rather than raising.
- Sample run shows honest grading: "Water boils at 100 degrees" scored 4/4, docked for omitting units and standard-atmosphere conditions.

## Retrieval metrics (pp. ≈168–174)
The gating question: *is the pipeline fetching the most relevant chunks for the query?* All classical metrics below require **golden chunks** — human-curated relevant chunks per query. The book is blunt: generating/maintaining these at production scale is "notoriously difficult, and often infeasible" because ground truth obsoletes as source data evolves.

### Set metrics (need golden chunks)
- **precision@k = |relevant chunks in top-k| / k** — signal-to-noise of what's passed to the generator; crucial when context window is scarce.
- **recall@k = |relevant found in top-k| / |all relevant chunks|** — coverage; how much you missed.
- **F1@k** — harmonic mean of the two, for when both matter equally.
- Precision–recall trade-off: trivially max recall by returning everything (precision craters); high precision by returning few confident chunks (recall craters). Optimize both simultaneously.

### Rank-aware metrics (need golden chunks + relevance ordering)
Motivated by rank 1 being worth more than rank 10 ("lost in the middle"). Notable footnote: lost-in-the-middle hurts *humans* reviewing ranked lists more than it hurts the generative LLM, which tolerates irrelevant chunks better than reading fatigue suggests.
- **MRR = mean over queries of 1/rank of first relevant chunk** (0 if none). Most interpretable; right choice when one good answer fast is the goal.
- **MAP:** average precision at each position where a relevant item appears (`AP_K = (1/N)·Σ Precision(k)·rel(k)`), then mean AP across queries. Penalizes burying relevant items; robust with multiple relevant chunks.
- **nDCG:** `DCG@K = Σ (2^rel_i − 1)/log2(i+1)`; `nDCG@K = DCG@K / iDCG@K` vs an ideal ranking. Unique advantage: **graded relevance** (0–3). Gold standard for complex retrieval flows, but overkill when you only serve top-3/5 chunks — MRR/MAP suffice there.
  - ⚠️ Erratum in source text: the recall@k formula is mislabeled as precision@k, and F1's denominator prints as P×R instead of P+R. Formulas above are the standard/correct forms.

### UMBRELA — reference-free chunk scoring (pp. ≈173–174)
Replaces golden chunks with LLM-as-a-judge per-chunk relevance on a 0–3 scale:
`0` irrelevant · `1` related but doesn't answer · `2` contains some answer, possibly buried · `3` dedicated to answering precisely.
- Prompt structure: intent analysis → match to likely intent (M) → trustworthiness (T) → final integer score (O), no reasoning emitted.
- Validated: scores correlate highly with human assessors (per UMBRELA paper); implemented in **Open RAG Eval**.
- Raw 0–3 scores can feed rank-aware metrics like nDCG. Recall remains impractical — computing it means scoring *every* chunk in the corpus.
- Cost model shift: human labeling effort ↔ compute cost + inference latency. Best for dynamic/large corpora where annotation can't keep up and budget allows API spend.
- Worked example (photosynthesis query): definitional chunk → 3; chlorophyll/chloroplast chunk → 2; mitochondria chunk → 0.

## Generation metrics (pp. ≈175–178)
Core question: *is the LLM using the provided chunks effectively to answer the query?* Keep most/all of these available in production.

| Metric | What it measures | Needs golden answer? | How computed |
|---|---|---|---|
| **Context utilization** | Does the generator use all necessary facts without distraction? | No | Inline-citation inspection (cheap/partial) or **AutoNuggetizer** |
| **Answer similarity** | Generated vs ground-truth answer semantics | **Yes** | BERTScore, ROUGE-L |
| **Answer relevancy** | Pertinence to question; penalizes incomplete/redundant answers | **Yes** | Similarity-style scoring vs golden answer |
| **Faithfulness / factual consistency** | Every statement verifiable from retrieved chunks; hallucination = extrapolation beyond them | No | HHEM hallucination-detection model, or LLM-as-a-judge — *the most critical generator metric* |
| **Citation accuracy** | Do citations actually support the statements they're attached to ("citation precision")? | No | Judge/verification of claim↔source support |
| **Response consistency** | Same query → same factual substance across repeated runs (LLMs are nondeterministic even at temp 0; wording drift is OK, facts must be stable) | No | N-run repetition + comparison |

**AutoNuggetizer pipeline** (from TREC nugget evaluation; in Open RAG Eval):
1. **Nugget generation & classification** — extract atomic facts from query+chunks; tag each `vital` (a good answer must contain it) or `ok` (nice-to-have detail).
2. **Support judgment** — for each generated answer, classify coverage of every nugget: `supported` / `partially supported` / `not supported`.
3. **Aggregation** — score and combine judgments into an overall response evaluation.
This is the tool that catches context-utilization failures (ignored chunks) that similarity metrics can't see.

**Bias & safety screening:**
- Dedicated safety classifiers in the eval loop: **Llama Guard** (text classifier over prompts + outputs; flags hate speech, sexual content incl. minors, self-harm, terrorism, incitement to violence) and **ShieldGemma** (multimodal text+images under customizable policies — relevant when RAG output mixes images and text). Findings feed back into retrieval filtering or refusal behavior.
- **Red teaming** complements automation with human adversaries crafting nuanced prompts (cultural context, ethical dilemmas) to surface what metrics miss — e.g., corpus over-reliance on a single perspective, subtle stereotype reinforcement. Successful probes become safety filters, prompt changes, or data-diversification work items.

## Evaluation platform survey (pp. ≈179–182)
Trade-off axis: flexibility/control (OSS) vs ease/managed infra (commercial).

| Framework | Model | Workflow | Strengths | Limitations |
|---|---|---|---|---|
| **Open RAG Eval** (Vectara + U. Waterloo, OSS) | Reference-free metrics only — no golden chunks/answers needed; just a query list | YAML config (queries file, connector, metrics) → JSON results; UI app "Open Evaluation" (Fig. 6-1); connectors for Vectara/LangChain/LlamaIndex or manual JSON feed | UMBRELA retrieval scoring, AutoNuggetizer groundedness/context utilization, HHEM hallucination score, citation metric, plus a **consistency metric** (run N times, report mean/σ of any other metric); small, fully open, extensible codebase | Younger ecosystem; reference-free = judge-compute costs apply |
| **Ragas** (OSS) | LLM-as-a-judge across a large metric set (RAG + agentic) | Build HF dataset: question / answer / contexts (+ optional `ground_truth`) → `ragas.evaluate()` | LangChain/LlamaIndex integration; **synthetic test-set generation from your docs** to bootstrap evals (use with care — synthetic queries may not match real user distribution) | Internals hard to inspect → low scores are hard to diagnose; **no reference-free metrics** → golden-dataset burden remains |
| **DeepEval** (OSS) | "LLM eval as unit testing" — pytest-native | Define `LLMTestCase`s; assert with `deepeval.assert_test()` against metric thresholds | Feels like pytest → ideal for CI/CD regression gates; 14+ metrics (faithfulness, answer relevancy, contextual recall…); **G-Eval** for arbitrary custom criteria | Code-centric workflow less friendly to non-devs (implied) |
| **Amazon Bedrock** (managed) | Managed eval jobs; pick a strong judge FM (e.g., Claude) | Configure job for retrieval-only or full retrieve-and-generate assessment | Zero infra to maintain; context relevance/recall + faithfulness/correctness; built-in responsible-AI dims (**harmfulness, stereotyping, answer refusal**); integrates AWS Guardrails | Judge-model opex; least transparency into evaluation prompts |

## Human feedback as ground truth (pp. ≈183–184)
Automated metrics struggle with subtlety and intent; end-user signals close the gap.
- **Primary KPI:** `User Satisfaction Rate = thumbs_up / (thumbs_up + thumbs_down)`.
- **Instrument every interaction**, logging: unique interaction ID, user prompt, retrieved context (docs/chunks), generated answer, feedback, timestamp + metadata (user ID, session ID).
- Four analyses this unlocks:
  1. **Overall satisfaction rate** — the bird's-eye KPI.
  2. **Satisfaction by topic/category** — classify prompts (keywords/embeddings/classifier) to find weak areas (book's example: features at 95% vs billing at 60% ⇒ billing knowledge-base problem).
  3. **Correlational analysis** — do low faithfulness/relevance scores track 👎? If yes, your automated metric is a good proxy; if not, it's misleading.
  4. **Failure analysis** — mine the most-thumbed-down interactions, root-cause (bad retrieval / hallucination / formatting), prioritize fixes.
- Best practice framing: treat feedback as a *direct* evaluation source with a standing review-and-remediate process over the worst queries/topics.

## Production integration: measurement + tuning (pp. ≈184–188)
A mature framework must do two jobs: **measurement** (systematic quality monitoring — baseline pre-launch, regular post-launch, and after any component/data/config change) and **tuning** (experimenting over chunking, embeddings, retrieval algorithm, LLM choice, prompts to find the best combination). Two cycles:

### Offline evaluation = the tuning engine + CI gate
- Every component upgrade (embedding model, reranker, prompt) passes an **evaluation gate** before deploy: benchmark-dataset test with a minimum quality bar.
- Enforce a **no-regression policy** on critical metrics, e.g., faithfulness and retrieval relevance above threshold or improved vs previous release; **P95 latency not up more than 5%**.
- Keep a **living benchmark**: promote difficult/failed real-world queries into the test suite as scope grows.
- **Version benchmark datasets alongside code** for audit trail and drift prevention.

### Online evaluation = early-warning system
Challenges: no golden datasets (real queries unpredictable), cost/latency of judging every call (a frontier judge "can double your operational costs").
Recommended architecture:
- **Asynchronous out-of-band eval:** serve the user immediately; background worker logs query/context/answer and runs reference-free checks (UMBRELA, AutoNuggetizer) with zero added perceived latency.
- **Intelligent sampling:** evaluate ~**5–10%** of traffic — statistically sufficient.
- **A/B testing** as gold standard: route a slice of users to a challenger pipeline vs champion; direct answer to "does this change actually help the user?"
- **Evaluation flywheel:** async automated scores + real-time thumbs feedback → catch regressions in the wild → promote high-value failure cases back into the offline golden dataset.

### Operating the LLM judge itself
- Judge calls are a secondary model call = a tax on dev cycle; in production, batch or sample rather than judging all interactions.
- **Log judge inputs (query/context/answer) + full reasoning + score as system telemetry** so you can audit whether quality dropped or the judge "had an off day."
- **Version judge prompts and model pins** — unversioned judges cause *evaluation drift* ("your yardstick moved").

### System metrics: latency & uptime (pp. ≈187–188)
Quality scores are necessary but insufficient; slow/unavailable/expensive systems fail regardless of accuracy.
- **Latency:** track average + tails (P95/P99); decompose per component (retriever / reranker / LLM) for targeted optimization.
- **Throughput:** QPS for capacity planning under load.
- **Reliability:** uptime target ≥99.9%; error-rate spikes (HTTP 5xx, timeouts) point at LLM API, vector DB, reranker, or hallucination-model trouble.
- **Cost:** per-component spend (vector DB, embedding/retrieval, reranker, hybrid search, token-priced LLM API) **plus the secondary eval cost layer** from judge tokens. Run an explicit evaluation budget: prefer reference-free metrics where suitable and sample expensive evals. Also watch CPU/GPU/memory utilization.

## Chapter conclusion (p. ≈188–189)
Evaluation spans the whole pipeline — ingestion → retrieval → generation. The chapter's summary of its own contributions: classical + rank-aware retrieval metrics; generation faithfulness/utilization/accuracy/citation/consistency; reference-free UMBRELA & AutoNuggetizer; production concerns (latency, uptime, cost, real-world conditions); human feedback; safety/bias/red-teaming; platform landscape. Framed as a strategic lever: metrics must align to use-case goals (some need recall, others faithfulness+latency) via a layered approach — automated metrics + human feedback, continuously.

## Practitioner checklist distilled from this chapter
1. Log every interaction: prompt, retrieved context, answer, citations, feedback, timestamps/session IDs.
2. Fix ingestion first: entity IDs + versioning + purge-on-reingest; duplicate-version canary test; specialized parsers for tables/PDF/DOCX/PPTX with pipeline logging.
3. Tune retrieval before the generator; measure with precision/recall@k (+MRR/MAP at small k); adopt UMBRELA when golden chunks are infeasible.
4. Track faithfulness (HHEM or judge) on every release; add AutoNuggetizer for context-utilization gaps; custom judge rubric for answer relevance.
5. Wire offline evals into CI/CD with no-regression thresholds and versioned benchmark datasets; promote failed real queries into the set.
6. Online: async sampled (5–10%) judging, A/B challengers, satisfaction-rate KPI correlated against automated metrics.
7. Treat the judge as production code: log reasoning telemetry, version prompts/models, budget its tokens.
8. Dashboard quality alongside latency/tail-latency, QPS, uptime, error rate, and cost.

---
*Digest generated from full read of `ch6_eval.txt` (1336 lines). Page numbers approximate from running footers; formulas corrected where source text garbled them.*
