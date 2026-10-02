# Backlog

Living task list. Newest decisions at the top of each section. When a section
grows into real design work it graduates to `docs/superpowers/specs/` +
`docs/superpowers/plans/` and this file keeps only the pointer.

Last updated: 2026-07-28.

---

## P0 — bugs found

### 1. `ensure_collection()` is never called on the API ingest path — fixed

`ingest/run.py:163` (the CLI/eval path) is the **only** caller. The API path —
`app/documents.py` → arq → `ingest/worker.py:93` → `IncrementalIngestor.ingest_document`
→ `QdrantVectorStore.upsert` — never creates the collection. Point a fresh
Qdrant at the API and the first upload upserts into a collection that does not
exist; the document lands in `failed`.

No test could catch it: `tests/_fakes.py:44` and
`tests/test_pipeline_integration.py:54` both stub `ensure_collection` out. This
is the same class of defect as the arq `redis_settings` bug in #20 — production
wiring that the fixtures paper over.

- [x] Call `ensure_collection(embedder.dimension)` once at worker startup —
      done via a new arq `on_startup` hook (`ingest/worker.py`), which arq
      awaits before the poll loop starts, so a broken collection fails the
      worker process instead of silently upserting into nothing
- [x] Add a test that drives the real `QdrantVectorStore` against an empty
      Qdrant and asserts the first upload reaches `ready` —
      `tests/test_ingest_worker.py::test_qdrant_live_first_upload_reaches_ready`
- [ ] Audit the rest of the fixture-stubbed wiring for the same class of bug:
      `get_registry`, `get_blobs`, `get_parsers` in `app/documents.py` —
      audited, **not fixed**: `build_document_registry`, `build_blob_store`,
      and `build_parser_registry` are overridden in every test and have zero
      coverage anywhere; `doc_registry_backend` defaults to `"postgres"`, so
      `PostgresDocumentRegistry` construction is entirely untested. Same
      defect class as PR #20 and the P0 just fixed.

### 2. Chunking configuration is dead — fixed

`ingest/worker.py:84` calls `chunk_document(doc)` with no arguments, hardcoding
`max_tokens=256, overlap=32`. `Settings.chunk_overlap = 200` (`core/config.py:105`)
is read by nothing outside `tests/test_sp4_config.py`. Chunk size is not
configurable at all, on any path.

- [x] Add `chunk_max_tokens` to `Settings`, thread both knobs through
      `ingest/worker.py` and `ingest/run.py`
- [x] Decide whether the existing `chunk_overlap=200` default is intended —
      it is ~6x the hardcoded 32 currently in force, so honouring it silently
      changes retrieval behaviour and the eval baseline. Decision: keep 200 as
      shipped; see the P3 sparse-index write-amplification entry below for the
      measured cost of that choice.

### 3. `Settings.max_chunks_per_corpus` is dead config

`core/config.py:147` defines `max_chunks_per_corpus: int = 2000` but nothing
outside docs reads it — exactly the defect class just fixed for
`chunk_overlap` above. With `chunk_overlap` now actually in force, chunk
volume per document is roughly 4x what it was while the field went unused, so
nothing caps the now-larger per-corpus growth on any path.

- [ ] Either wire this into the ingest path (reject or warn past the cap) or
      remove the field if no cap is actually wanted

---

## P1 — pluggable parsing, structure-aware chunking, remote OCR

This is one sub-project, not three. The blocker for all of it is a single line:
`ingest/parsers/unstructured_parser.py:18` flattens parsed elements to
`"\n\n".join(str(el) for el in elements)`. The parser protocol returns
`list[Document]` — a text blob — so every structural signal a parser extracts is
destroyed before the chunker can see it. Structure-aware chunking is impossible
on top of that contract no matter which parser is plugged in.

### The unlock: a Block intermediate representation

```python
class Block(BaseModel):
    kind: Literal["heading", "paragraph", "table", "list_item", "caption", "page_break"]
    text: str
    level: int | None            # heading depth
    page_no: int | None
    bbox: tuple[float, ...] | None
    metadata: dict
```

Parsers emit `list[Block]`; chunkers consume `list[Block]`. Each of the three
asks then falls out of the same mechanism instead of needing its own.

- [ ] Define `Block` in `core/types.py` and a `BlockParser` protocol in
      `ingest/parsers/base.py`
- [ ] Keep `DocumentParser` working during the migration — a `blocks_to_document`
      adapter means the worker does not have to change on day one
- [ ] Port `PdfParser` (pypdf already exposes per-page text → free `page_no`)
- [ ] Port the docx path (python-docx exposes heading styles and tables directly)
- [ ] Port the html path (heading levels and `<table>` are in the DOM)

### Pluggable parser selection

- [ ] `parser_overrides` config: MIME → provider name, resolved through
      `build_parser_registry` exactly like `build_embedder` / `build_vector_store`
- [ ] Registry of named parsers so adding one is a class plus one entry
- [ ] Document the contract for third-party/custom parsers

### Structure-aware chunker

- [ ] `StructuralChunker`: never split across a heading boundary, keep tables
      whole regardless of token budget, emit the heading path
      (`"3. Warranty > 3.2 Exclusions"`) into `Chunk.contextual_prefix` — which
      feeds `embed_text` for free
- [ ] Carry `page_no` into chunk metadata so citations can cite a page
- [ ] `chunker` config knob; today's `chunk_document` stays as the `"token"`
      strategy and remains the default until the eval gate says otherwise
- [ ] Re-baseline the eval run — chunking changes retrieval, so this needs a
      gate comparison, not a vibe check

Note: today's chunk text is `encoder.decode(tokens)` (`ingest/chunking.py:117`),
a token-boundary reconstruction rather than a substring of the source. No
character offsets are retained, so a chunk cannot currently be mapped back to a
page or bbox. The Block IR is what makes that possible.

### Remote OCR

Hosted elsewhere, called over HTTP — no local models, no weights in the image,
nothing downloaded at runtime.

- [ ] `RemoteOcrParser`: POST bytes to a configured endpoint, receive blocks
- [ ] Config: endpoint URL, API key, timeout, max pages, and which MIME types
      route to it
- [ ] Fail-closed and bounded: an OCR outage must mark the document `failed`
      with a clear error, never hang the worker or half-ingest
- [ ] Decide the routing rule for PDFs — probably "try pypdf first, fall back to
      OCR when the text layer is empty", which is exactly the case
      `PdfParser` currently raises `ParserError` on

---

## P2 — drop `unstructured`

Deferred, not rejected. Raised on 2026-07-28 and left open so the Docker work
would not prejudge it.

`unstructured` is used for exactly two formats (docx, html) and exactly one
line of behaviour (concatenate element text). It costs, measured inside the
built image:

| package | size |
|---|---|
| llvmlite | 169 MB |
| spacy | 125 MB |
| numba | 35 MB |
| thinc | 16 MB |
| en_core_web_sm | 15 MB |
| lxml | 13 MB |

≈ **375 MB of a 666 MB venv**. `numba`/`llvmlite` are pulled in by
`unstructured` alone (`uv tree --package numba --invert`).

It also downloads and installs an `en_core_web_sm` wheel into `site-packages`
on the first docx/html parse. That is defused *in the image* (the model is baked
in at build time, and the build asserts it loads), but the download path still
fires for host runs, `make api`, and CI.

Replacing it with python-docx + beautifulsoup4 is ~50 lines. It also folds
naturally into the P1 Block work, since both parsers have to be rewritten to
emit blocks anyway — doing it twice would be wasted effort.

- [ ] Decide: fold the unstructured removal into the P1 parser rewrite (likely),
      or do it standalone first
- [ ] If removed, drop the `en_core_web_sm` bake step from the Dockerfile

---

## P3 — operational

- [ ] **Branch protection on `main`** — still none (`gh api .../branches/main/protection`
      → 404), so no check has ever been *required*. Now that CI is trustworthy
      again (repaired in #22), require `Lint & Offline Tests`,
      `Live Database ACL Isolation Tests` and `Gated Status Check`
- [ ] **CI secrets** — `NVIDIA_API_KEY` and `LANGFUSE_*` are unset, so the eval
      job skips on every PR and each one needs the `eval-skip-approved` label by
      hand. Until then the eval gate catches zero regressions
- [ ] **Langfuse baseline run** — the gate compares against a run named
      `baseline` for the `hotpotqa` dataset. It does not exist yet, so the gate
      could not pass even with secrets
- [ ] **End-to-end smoke test on merged `main`** — the pieces are verified
      individually (console via uvicorn+curl, arq fix against a real
      `redis:8-alpine`, the image against a real daemon) but the full
      upload → status → query loop has never been driven against live infra
- [ ] **Sparse index write amplification** — `TenantSparseStore.add`
      (`providers/sparse/tenant_store.py:42-56`) re-pickles a tenant's *entire*
      chunk list on every document write. O(tenant corpus) per upload; fine at
      demo scale, a real bottleneck later. `chunk_overlap=200` going live
      (this branch, P0 #2) multiplies the constant ~3.9x — pre-existing
      complexity, not a new bug, but it compounds against an already
      quadratic curve (bytes written grow O(N²) in upload count). Measured on
      20 sequential uploads of a 4,000-word doc, one tenant:
      overlap=32 → 980 chunks / 0.90 MB pickle / 0.37s;
      overlap=200 → 3,860 chunks / 3.55 MB / 0.98s. `run_ingest` is
      synchronous inside an async arq job, so this window also blocks the
      event loop and stalls arq's poll heartbeat ~4x longer
