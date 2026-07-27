"""CLI entry point for the ingest pipeline.

Usage
-----
    python -m ingest.run --dataset hotpotqa [--contextual] [--limit 500]

NOTE: This module makes real network calls (HuggingFace downloads, embedding
API, vector store upsert). It is NOT exercised by the offline test suite.
All heavy imports and registry calls are inside main() so importing the module
is safe.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from core.types import Document, Chunk
from core.config import Settings
from core.interfaces import PIIDetector
from ingest.audit import PIIAuditLog


def _resolve_adapter(name: str):
    """Return a DatasetAdapter instance for the given corpus name."""
    if name == "hotpotqa":
        from corpora.hotpotqa.adapter import HotpotQAAdapter
        return HotpotQAAdapter()
    if name == "arxiv":
        from corpora.arxiv.adapter import ArxivAdapter
        return ArxivAdapter()
    if name == "financebench":
        from corpora.financebench.adapter import FinanceBenchAdapter
        return FinanceBenchAdapter()
    raise ValueError(f"Unknown dataset: {name!r}")


def _apply_pii_ingest_policy(
    docs: list[Document],
    settings: Settings,
    detector: PIIDetector,
    audit: PIIAuditLog
) -> tuple[list[Document], list[Chunk]]:
    from ingest.pii import redact

    if settings.pii_mode == "redact":
        # Redact raw docs prior to chunking
        clean_docs = []
        for doc in docs:
            spans = detector.detect(doc.text)
            # Record audit values
            audit.record(
                tenant_id=doc.tenant_id,
                doc_id=doc.doc_id,
                chunk_id="doc_redaction",
                text=doc.text,
                spans=spans
            )
            clean_text = redact(doc.text, spans)
            clean_docs.append(doc.model_copy(update={"text": clean_text}))
        return clean_docs, []
    
    else:  # keep mode
        return docs, []


def _chunk_documents(docs: list[Document], settings: Settings) -> list[Chunk]:
    """Chunk every document using the token budget from Settings, so a config
    change (chunk_max_tokens / chunk_overlap) actually changes the chunks this
    CLI/eval path produces, same as the API/worker path."""
    from ingest.chunking import chunk_document

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(
            doc,
            max_tokens=settings.chunk_max_tokens,
            overlap=settings.chunk_overlap,
        ))
    return chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the RAG stack.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "arxiv", "financebench"],
    )
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    # CLI mode override
    parser.add_argument(
        "--pii-mode",
        choices=["redact", "keep"],
        default=None,
        help="Overwrite settings.pii_mode for this run"
    )
    args = parser.parse_args(argv)

    # --- Setup config & dependencies ---
    from core.config import get_settings
    settings = get_settings()
    
    # Overwrite if passed
    if args.pii_mode:
        settings = settings.model_copy(update={"pii_mode": args.pii_mode})

    # Load documents
    adapter = _resolve_adapter(args.dataset)
    docs = adapter.load(limit=args.limit)
    print(f"[ingest] loaded {len(docs)} documents from {args.dataset}")

    # Build logger/detector tools
    from core.registry import build_pii_detector
    from ingest.audit import PIIAuditLog
    detector = build_pii_detector(settings)
    audit = PIIAuditLog(settings)

    # Apply policy
    try:
        clean_docs, _ = _apply_pii_ingest_policy(docs, settings, detector, audit)
    except Exception as e:
        print(f"[ingest] PII detection or audit failed: {e}. FAILING CLOSED: aborting.")
        raise e

    # --- Chunk redacted or raw documents ---
    all_chunks = _chunk_documents(clean_docs, settings)
    print(f"[ingest] produced {len(all_chunks)} chunks")

    # In keep mode, tag PII on generated chunks and write audit log
    if settings.pii_mode == "keep":
        for chunk in all_chunks:
            try:
                spans = detector.detect(chunk.text)
                if spans:
                    # tagging
                    chunk.metadata["pii_types"] = sorted({s.type for s in spans})
                    # audit records (must succeed in keep mode to allow storing)
                    audit.record(
                        tenant_id=chunk.tenant_id,
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        spans=spans
                    )
            except Exception as e:
                print(f"[ingest] PII validation failed for chunk {chunk.chunk_id}: {e}. FAILING CLOSED.")
                raise e

    # --- Optional contextual prefixing ---
    if args.contextual:
        from core.registry import build_generator
        from ingest.contextual import ContextualPrefixer

        generator = build_generator(role="context", settings=settings)
        prefixer = ContextualPrefixer(generator, cache_dir=settings.contextual_cache_dir, settings=settings)
        doc_by_id = {d.doc_id: d.text for d in clean_docs}
        all_chunks = prefixer.annotate(all_chunks, doc_by_id)
        print(f"[ingest] contextual prefixes applied to {len(all_chunks)} chunks")

    # --- Embed ---
    from core.registry import build_embedder, build_vector_store, build_sparse_retriever

    embedder = build_embedder(settings)
    texts = [c.embed_text for c in all_chunks]
    print(f"[ingest] embedding {len(texts)} texts ...")
    vectors = embedder.embed_documents(texts)
    chunks_with_embeddings = [
        c.model_copy(update={"embedding": v}) for c, v in zip(all_chunks, vectors)
    ]

    # --- Dense store ---
    vector_store = build_vector_store(settings)
    vector_store.ensure_collection(embedder.dimension)
    vector_store.upsert(chunks_with_embeddings)
    print(f"[ingest] upserted {len(chunks_with_embeddings)} chunks to vector store")

    # --- Sparse store (BM25, persisted) ---
    sparse = build_sparse_retriever(settings)
    sparse.index(all_chunks)
    cache_dir = Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = cache_dir / f"bm25_{args.dataset}_{settings.vector_store}.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(sparse, f)
    print(f"[ingest] BM25 index saved to {bm25_path}")

    # --- Export golden eval set (shape consumed by eval.dataset_cli seed) ---
    import json

    golden = adapter.build_golden(limit=args.limit)
    eval_dir = Path("data") / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    golden_path = eval_dir / f"{args.dataset}.json"
    with golden_path.open("w") as f:
        json.dump(
            [
                {
                    "question": g.question,
                    "ground_truth": g.answer,
                    # doc-level gold ids; pipeline.run returns doc-level retrieved_ids
                    "relevant_chunk_ids": list(g.relevant_doc_ids),
                    "tenant_id": g.tenant_id,
                }
                for g in golden
            ],
            f,
            indent=2,
        )
    print(f"[ingest] wrote {len(golden)} golden items to {golden_path}")


if __name__ == "__main__":
    main()
