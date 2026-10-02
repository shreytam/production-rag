# Engineering Digest — Chapter 2: The Base RAG Stack

**Book:** *Hands-On RAG for Production* (Mendelevitch & Bao)
**Source:** `ch2_base_stack.txt` · **Coverage:** book pages 24–62 (full chapter; file ends with the first paragraph of Ch. 3 "Scaling Your RAG Stack", p. 63)

---

## TL;DR

Chapter 2 defines the minimum viable RAG pipeline as five layers — **parsing → chunking → embedding → vector indexing/search → LLM generation** — organized into two flows:

- **Ingestion flow** (async, with retries): parse sources into text, chunk to fit model constraints, embed chunks, store in a vector database.
- **Query flow**: rewrite the user query into a *retrieval query* → semantic search over embeddings → rerank → feed chunks + **original** user query to a generative LLM.

Core engineering messages: ingestion errors propagate downstream and can't be fixed later; chunk size is bounded by the **embedding** model's window (often 8k tokens), not the LLM's; dot-product retrieval needs reranking because it ignores cross-attention between query and chunk; ANN search (HNSW) trades exactness for orders-of-magnitude speed; and no layer succeeds in isolation — quality is determined by the interactions across the whole stack.

---

## 1. Architecture: The Two Flows (pp. 24–27)

```
INGESTION (async, orchestrated with retries)          QUERY
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ sources  │→ │ parsing  │→ │ chunking  │→ │ embedding│→ [vector DB]
└──────────┘   └──────────┘   └───────────┘   └──────────┘        │
                                                                   ▼
user query → query rewriting → retrieval (semantic) → reranking → generative LLM → response
```

### Ingestion flow (4 steps: parsing, chunking, embedding, indexing)
- **Sources:** databases, files (local/cloud), APIs, web scraping. DB/API data is structured (schema known in advance) → minimal preprocessing, e.g., flatten LLM–user conversations into a 2D table (`text, role, timestamp, conversation_id`); only message text gets embedded.
- **Files** are harder: formats (PDF/DOCX/PPTX/HTML/Markdown) mix knowledge text with styling/typesetting info; scanned images require OCR.
- **Why not skip chunking/embedding:** raw character-space matching fails across surface forms ("United States" vs "USA", "Silicon Valley" vs "硅谷"/"矽谷"). Embeddings map text to vectors that capture semantics; keyword/inverted indexes complement them for unseen proper nouns and typo tolerance (deferred to Ch. 3 "Hybrid Search", p. 76). All representation/indexing methods fall under **indexing**.

### Query flow (Figure 2-2)
1. **Query rewriting** — split user intent into *command* vs *context*. Example: "Write a poem based on my travels in 2025" → retrieval query = `"my travels in 2025"` (including "write a poem" would retrieve poems instead of travel facts).
2. **Retrieval** — semantic search (dense): embed query, dot-product against stored vectors. Modern embeddings are unit-normalized, so dot product == cosine similarity. Keyword search (sparse: TF-IDF, BM25) operates in lexicon space; modern models also emit **sparse vectors** for term matching. Base stack = pure semantic search only.
3. **Reranking** — two stated reasons embedding ranking isn't optimal:
   - Retrieved chunks are often redundant → **MMR** (maximal marginal relevance, from 1998) balances relevance with diversity to save token budget.
   - Dot product scores texts *independently*; transformer-based **cross-attention rerankers** evaluate query+chunk jointly and capture nuanced relevance.
4. **Generation** — chunks + original user query go into an LLM prompt (§6).


---

## 2. Document Parsing (pp. 28–37)

**Why it matters:** "accuracy at this stage is paramount" — ingestion errors compromise the whole knowledge base and can't be repaired at retrieval/generation time.

### Format realities
- **PDF** (rooted in PostScript) renders *visual appearance, not logical structure*. A word is stored as individual characters + 2D coordinates; only **tagged PDFs** (accessibility) carry structure. Reconstructing words/sentences/columns requires coordinate clustering/alignment algorithms — otherwise two-column pages get interleaved into gibberish.
- **Scanned PDFs** are images per page → OCR. OCR accuracy depends on image resolution/clarity, font complexity/style, and layout. Tools: Tesseract (OSS), Google Cloud / Azure OCR APIs, startups (e.g., Reducto.ai).
- **DOCX/HTML** are XML-family formats: structured, easier to parse; strip presentation tags (`<b>` etc.) but **don't discard everything** — some attributes are semantic signal.

### Metadata is a first-class citizen (Table 2-1)
Chat-log example: keep `class="user"/"AI"` as metadata `{"who": "user" | "AI"}` alongside text. At query time, rewriting splits the question into a semantic query + a **metadata filter** ("who = AI"). Filtering is a plain database operation (SQL `where`-like) — *not* part of semantic search.

### VLM-based parsing (pp. 30–31)
Vision–language models (e.g., GPT-5.x series) can parse complex layouts (slides, forms, infographics) directly — but: hallucination risk, often can't reproduce embedded images, expensive and slow at enterprise scale. **Recommendation:** "cheapest successful parser-first" strategy (PDF libs → HTML parsers → OCR), reserving VLMs for difficult high-value documents or as fallback when classical parsers fail.

### Code walkthrough (pp. 31–36)
| Task | Library / API | Key API points |
|---|---|---|
| PDF text | PyMuPDF | `page.get_text()` mixes table+prose; `page.get_text("blocks")` groups by block but still can't label tables |
| PDF tables | PyMuPDF | `page.find_tables()` + `table.extract()` → 2D list; use `table.bbox` with `page.get_text(clip=bbox)` then line-filter to get non-table prose (`extract_unstructured_text()`) |
| PDF images | PyMuPDF | `page.get_images()` returns *pointers* (xrefs), not bytes; `doc.extract_image(xref)` yields actual image data |
| DOCX text/tables | python-docx | `document.paragraphs`, `document.tables[].rows[].cells` |
| DOCX images | stdlib `zipfile` | DOCX = ZIP of XML + media; scan `namelist()` for `.jpg/.jpeg/.png/.gif` |
| LLM parsing | OpenAI GPT-5.1 | upload via `client.files.create(purpose="user_data")`, then `chat.completions.create` with a `type:"file"` content part + plain-English instruction ("extract text excluding tables/images"; "extract tables, return Markdown") |

Other named tools: pypdf (PyPDF2/3/4 lineage), pdfminer.six, Adobe PDF Extract API, Unstructured.io, Beautiful Soup.


---

## 3. Text Chunking (pp. 37–42)

### Why chunk (four reasons given)
1. **Finite LLM context window** — even 400k–1M tokens is "deceptive abundance": 400k tokens ≈ 42 hours of continuous speech (120 wpm × ~1.3 tokens/word) — not enough to hold all earnings-call transcripts for a market-consensus question. Chunking maximizes *relevant* density in the window.
2. **Inference latency/cost** — TTFT grows roughly **quadratically** with context length (Table 2-3, Llama 3.3 70B, 2×H100, FP8):

   | Tokens | TTFT |
   |---|---|
   | 200 | 31 ms |
   | 500 | 47 ms |
   | 1,000 | 82 ms |
   | 5,000 | 406 ms |
   | 10,000 | 1,833 ms |

   Cost framing: single-H100-class AWS instance ≈ $6.88/hr ≈ $5,022/mo; the 2×H100 config behind those numbers would exceed $10k/mo.
3. **Embedding models have much smaller windows** — SOTA embedders: 8k tokens (OpenAI text-embedding-3-*, Gemini gemini-embedding-002); chunks must respect the *embedder's* limit or get silently truncated.
4. **Response quality** — a multi-topic chunk yields a weakly-blended embedding per topic; smaller focused chunks retrieve better and present less noise at generation. LLM reasoning also degrades as context length grows (per Anthropic's 1M-context GA data).

### Strategy comparison (Table 2-4)
| Strategy | Pros | Cons |
|---|---|---|
| **Fixed-size** (+ optional overlap) | Simple, predictable size, efficient for indexing/batching | Breaks sentences/semantic units; needs overlap → redundancy |
| **Sentence/paragraph** (content-aware) | Preserves linguistic structure; easy via NLP tools (spaCy/Stanza/NLTK sentencizers) | Boundary detection error-prone (abbreviations); uneven sizes |
| **Recursive** (e.g., LangChain `RecursiveCharacterTextSplitter`) | Flexible delimiter hierarchy (paragraphs→sentences→clauses) | Complex to tune; delimiter-dependent; still ignores semantics |
| **Document-structure** | Follows heading hierarchy (`# → ## → ###`), preserves logical organization | Needs well-structured docs; format-specific |
| **Semantic** (cluster sentences by topic/similarity) | Semantically coherent chunks; aligns with embedding search | Expensive (embeddings+clustering); unpredictable sizes; hard to debug |

### Evaluation guidance
- Retrieval stage: BEIR-style benchmarks; precision/recall/F1/nDCG/MRR. Generation stage: QA/MRC benchmarks; BERTScore or LLM-judgment.
- Beware confounds: **"lost in the middle"** — LLMs attend most to chunks at the start/end of context, so generation-stage scores can mislead strategy comparisons.
- Empirical finding cited: Qu, Bao & Tu (EMNLP 2024), *"Is Semantic Chunking Worth the Computational Cost?"* — fixed-size vs semantic chunking made **no difference on BEIR/RAGBench**. Caveat: those benchmarks are pre-RAG-era with short passages; long-text benchmarks may change the verdict.
- Practice: evaluate on your own dataset (Chroma has a reference example).

### Code notes (p. 42)
- spaCy `en_core_web_sm` + `doc.sents`: handles "Mr.", "A.I.", mid-sentence "?" correctly where naive punctuation splitting fails.
- Fixed-length slicing with overlap (30-char window, 8-char step overlap): visibly breaks words ("He teac"), but impact of that noise shrinks as chunks grow — fixed-size stays attractive for speed.


---

## 4. Embedding Models (pp. 43–48)

### Concept
Embedding = text → vector of floats such that **representation similarity aligns with semantic similarity**. Solves NLP's long-standing misalignment of surface vs semantic (dis)similarity: "Uncle Sam" ↔ "US Government" (same meaning, different surface) and "cat" vs "hat" (similar surface, unrelated). Not just synonyms: with the right embeddings, query "cases that happened in the US" can retrieve California cases while rejecting Alberta (Canada) ones. Terminology: dense representation / **dense retriever**; strictly, *tokens* are embedded (subword units like "geo"+"political").

### Brief history
- **Elman network** (1990): RNN-era, one-hot vectors (vocab-sized, single 1).
- **Word2Vec** (Mikolov, Google, 2013): skip-gram (predict surrounding words); famous `king − man + woman ≈ queen`; **static** embeddings (fixed per word after training).
- **BERT** (Google, 2018): **contextualized/dynamic** embeddings — a token's vector depends on its context.

### Selection criteria
- **Size vs performance:** use the **MTEB leaderboard** to find the balance point.
- **Dimensions:** higher = more nuance but more compute/storage. Many APIs support truncation via **Matryoshka Representation Learning (MRL)** — loss = sum of losses at multiple dimensionalities, pushing discriminative info into low dims. Practice: truncate at power-of-two fractions (1/8, 1/4, 1/2), since models are optimized at those granularities.
- **Context window** (tokens, at writing): Qwen3-Embedding-{0.6B,4B,8B} **32k**; OpenAI text-embedding-3-{small,large} **8k**; gemini-embedding-002 **8k**; BAAI BGE-M3 **1k**.

### Practical tips (p. 46)
- Match chunk size ≤ embedder window; Transformers-style frameworks **silently truncate** otherwise → silent retrieval degradation.
- Respect your vector DB's dimension cap — pgvector caps full-precision 32-bit float vectors at **2,000 dims**.
- Lower precision (fp16 / int8) saves storage & compute; most endpoints accept a precision parameter.
- Transport: JSON is wasteful for numerals (`123` = 1 byte binary, 3 bytes as string; float strings risk precision loss). Prefer **Base64-encoded vectors** (full precision, smaller payloads) with client-side decode.

### Code experiment (pp. 46–48)
`sentence-transformers` with `all-MiniLM-L6-v2` (384-dim): four sentences ("happy" / "joyful" / "pessimistic" / "not optimistic") → cosine similarity matrix shows happy↔joyful = 0.8151, pessimistic↔not-optimistic = 0.7047, cross-pairs ≈ 0.34–0.52. Random-subspace experiment: computing similarities on an arbitrary slice of dimensions reproduces the same structure — modern embedding spaces are so redundant that random subspaces still carry usable semantics (a nice intuition pump for MRL truncation).


---

## 5. Vector Databases & Vector Search (pp. 49–54)

### Similarity & search complexity
- Normalized embeddings → **dot product == cosine similarity** (magnitude influence removed; pure angular comparison).
- Brute force over all vectors = O(n) — fine for small data kept in RAM (**exact nearest neighbors / ENN / FlatIndex**), not for millions–billions of chunks.

### ANN: HNSW (Malkov & Yashunin, IEEE TPAMI 2020)
Multi-layer graph: each vector links to nearby vectors in its layer; top layers are sparse/coarse, bottom layers dense/fine. Query descends greedily, always moving toward the query. The book's analogy: find the nearest US city to Los Altos — compare against 2 landmark cities (LA/NYC) → western big cities → Bay Area cities → South Bay cities; a handful of distance computations instead of every city in the country.
- "Approximate": greedy descent can land in the wrong region near boundaries → returns a *near*-nearest neighbor, rarely exact.
- In practice accuracy ≈ exact search at a fraction of the cost. Reference implementation: **Meta/Facebook's FAISS**.

### What makes a *vector database* (vs a bare index)
Persistence, updates/deletes, metadata management, filtering, scalability, integrations. Specialized DBs emerged first (proprietary **Pinecone**; OSS **Milvus**, **Weaviate**, **Qdrant**) but general-purpose stores now do vector search too: **pgvector** (PostgreSQL), **sqlite-vec** (SQLite), **Atlas Vector Search** (MongoDB).

### Metadata filtering
Filters run on metadata fields (columns), not embeddings — but placement matters: pre-filter narrows the search space before expensive vector search (e.g., company + quarter in financial reports). Options: pre-, parallel-with, or post-filtering (Pinecone has a good blog); **ACORN** jointly performs filtering + vector search.

### Tuning `k` and dimensions
- **k** (results returned): too small → answer ranked beyond k is missed; too large → context overflow, noise degrading generation, higher cost/latency.
- MRL-truncated dims shrink the DB and speed up search, but over-truncation degrades quality — balance per use case and model; mind the DB's max dimension support.

### pgvector walkthrough (pp. 53–54)
1. `CREATE EXTENSION IF NOT EXISTS vector;`
2. Table: `sentence_embeddings(sentence TEXT, embedding VECTOR(384))` (384 = all-MiniLM-L6-v2 dim).
3. Index: `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)`.
4. Insert NumPy rows as float lists with `%s::vector` cast.
5. Search: `SELECT text, 1 - (embedding <=> :q) AS similarity ... ORDER BY embedding <=> :q LIMIT k`. `<=>` returns cosine *dissimilarity* in pgvector → complement it for similarity.
6. Demo results behave as expected ("smiling person" ≈ happy/joyful ≈0.69–0.76, pessimistic-style sentences drop to ~0.38).

⚠️ Two code issues worth verifying before reuse (see §10 Errata): the INSERT/SELECT references column `text` while the table defines `sentence`, and the HNSW index uses `vector_l2_ops` while queries order by `<=>` (cosine distance), which pairs with `vector_cosine_ops`.


---

## 6. Generative LLM Layer (pp. 55–61)

### Role & selection
- Generation is a **text-to-text transform**: you send the *original text chunks* (not embeddings) plus the original user query. Dominant tasks: **summarization** and **question answering** — both heavily represented in LLM training data.
- **Speed vs quality:** bigger models win on complex tasks but are often on par on simple ones; build an eval set and pick the cheapest model meeting your bar.
- **Inference speedups** (especially self-hosted/on-prem/air-gapped): **quantization** (fp32 → fp16/bf16 → int8 → int4), **FlashAttention** (memory-efficient attention), serving frameworks (**vLLM**, **Ollama**).
- **Data privacy:** air-gapped deployments rule out proprietary HTTP APIs → open weights. Rule of thumb: an OSS LLM **>70B parameters** is "decent enough" for summarization/QA (gap with proprietary is small at writing).

### Prompt engineering for RAG (pp. 57–58)
- Use the **original user query**, not the rewritten retrieval query, in the generation prompt.
- Put the **query before the context** — most LLMs perform better that way given training data.
- Add a CoT nudge: *"If the answer is not obviously present… think step by step…"*
- Anti-hallucination clause: *"If an answer cannot be reasonably inferred from the context, please simply say 'I don't know.' If you used any assumptions… clearly state what assumptions are made."*
- Task-specific templates (QA vs search-style summarization); persona/domain background ("you are an expert in science/sports") can help.
- **XML tags** (`<query>`, `<context>`) to delineate prompt parts.

### Evaluation of LLMs & prompt templates (pp. 58–59)
1. Build eval queries mirroring real usage (reuse logs, or synthesize via LLM using knowledge of data + user behavior).
2. Build critics, two standard aspects: **relevance** (does it answer?) and **faithfulness** (supported by retrieved docs?).
   - **LLM-as-a-judge:** flexible, same LLM can generate and judge; drawbacks — slow/expensive, hallucinating/flawed reasoning, inconsistent across runs.
   - **Dedicated NLG evaluation models:** smaller, fine-tuned judges — faster, cheaper, more robust. Examples: Vectara's **HHEM** (faithfulness; 5M+ downloads Aug 2024–Aug 2025), Galileo's **Luna**.
   - **Human evaluation:** the "golden" method — most accurate/nuanced, most expensive; use as final small-scale check after automated passes.
3. Multi-criteria scoring: weighted score system; pick the best-performing (LLM × template) combo.

### Demo (pp. 60–61)
`generative_LLMs.ipynb`: chapter text as context + 3 questions answered by Claude Sonnet 4.5 via `anthropic.Anthropic().messages.create(model="claude-sonnet-4-5", ...)` with the simple template above — all three answers judged relevant and faithful.

---

## 7. Key Numbers Cheat Sheet (as stated "at the time of writing")

| Item | Value |
|---|---|
| LLM context windows | GPT-5.1: 400k · Claude Sonnet 4.5: 1M · Gemini 3: 1M |
| Embedding context windows | Qwen3-Embedding (0.6B/4B/8B): 32k · OpenAI text-embedding-3-{small,large}: 8k · gemini-embedding-002: 8k · BGE-M3: 1k |
| Example embedder dim | all-MiniLM-L6-v2 = 384 |
| pgvector dim cap | 2,000 dims (32-bit full precision) |
| TTFT, Llama 3.3 70B / 2×H100 FP8 | 200 tok → 31 ms … 10k tok → 1,833 ms (~quadratic) |
| H100-class AWS cost | $6.88/hr ≈ $5,022/mo (2×H100 would exceed $10k/mo) |
| Speech math | 120 wpm × ~1.3 tokens/word → 400k tokens ≈ 42 h of speech |
| OSS LLM sizing rule | >70B params ≈ decent for summarization/QA |
| MMR origin year | 1998 |
| HHEM adoption | 5M+ downloads (Aug 2024 – Aug 2025) |

## 8. Production Lessons & Pitfalls

1. **The ingestion bottleneck** — poor OCR or lost formatting propagates downstream; retrieval/generation can't reliably repair it. Invest at parse time ("cheapest successful parser-first"; VLM as fallback).
2. **Naive chunking is rarely sufficient** — validate chunking via retrieval evaluation on *your* data; don't trust generation-stage scores alone ("lost in the middle" confound).
3. **Retrieval can still fail** with great embeddings — redundancy, noise, scope mismatch → motivates hybrid search + filtered retrieval (Ch. 3).
4. **Chunk size is bounded by the embedder window, not the LLM window** — silent truncation in serving frameworks means you may never see the failure.
5. **Metadata early** — capture role/date/source attributes during parsing; they enable cheap pre-filtering that beats semantic search on precision for scoped questions.
6. **Rerank before generating** — embedding similarity ≠ LLM-optimal ordering (redundancy + no cross-attention).
7. **Tune k deliberately** — misses below k, noise/cost above it.
8. **No layer succeeds alone** — chunking quality gates retrieval, retrieval gates generation, index design drives latency/cost; optimize end-to-end.

## 9. Code Artifacts Inventory (chapter's GitHub notebooks referenced)

- `pymupdf` parsing: page text / blocks / tables (`find_tables`, `bbox` clip filtering) / images (`get_images` + `extract_image(xref)`)
- `python-docx` text+tables; stdlib `zipfile` image extraction from DOCX
- OpenAI GPT-5.1 file-upload parsing (text excluding tables/images; tables → Markdown)
- spaCy sentencizer chunking; fixed-length chunker with overlap
- `sentence-transformers` embeddings + cosine similarity matrix + random-subspace experiment
- `pgvector-simple.ipynb`: extension setup, VECTOR(384), HNSW index, insert, `<=>` search
- `generative_LLMs.ipynb`: Claude Sonnet 4.5 RAG answering with prompt template

## 10. Errata / Verify-Before-Reuse (spotted while reading)

1. **pgvector column mismatch:** table is created as `(sentence TEXT, embedding VECTOR(384))`, but INSERT/SELECT statements reference a column named `text`. As printed, the INSERT would fail.
2. **HNSW opclass mismatch:** index built `USING hnsw (embedding vector_l2_ops)` but queries `ORDER BY embedding <=> ...` (cosine distance). For the index to serve those queries, use `vector_cosine_ops`.
3. Minor typos in prose: "pgector" (p. 53), duplicated bullet markers around context-window lists.

## 11. Cross-References / What's Deferred

- **Ch. 3 (Scaling Your RAG Stack):** hybrid search (p. 76), advanced reranking, ingestion at scale, guardrails, hallucination handling, UX.
- **Ch. 6:** full RAG evaluation methodology.
- **Ch. 8:** multimodal content (tables/images inside documents).
- **Ch. 4:** data security & privacy for scaled production.

---

*Digest generated from full read of `ch2_base_stack.txt` (1,621 lines; chapter proper ends p. 62).*
