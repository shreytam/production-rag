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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the RAG stack.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "arxiv", "financebench"],
    )
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    # --- Load documents ---
    adapter = _resolve_adapter(args.dataset)
    docs = adapter.load(limit=args.limit)
    print(f"[ingest] loaded {len(docs)} documents from {args.dataset}")

    # --- Chunk ---
    from ingest.chunking import chunk_document

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"[ingest] produced {len(all_chunks)} chunks")

    # --- Optional contextual prefixing ---
    if args.contextual:
        from core.registry import build_generator
        from core.config import get_settings
        from ingest.contextual import ContextualPrefixer

        settings = get_settings()
        generator = build_generator(role="context", settings=settings)
        prefixer = ContextualPrefixer(generator, cache_dir=settings.contextual_cache_dir)
        doc_by_id = {d.doc_id: d.text for d in docs}
        all_chunks = prefixer.annotate(all_chunks, doc_by_id)
        print(f"[ingest] contextual prefixes applied to {len(all_chunks)} chunks")

    # --- Embed ---
    from core.registry import build_embedder, build_vector_store, build_sparse_retriever
    from core.config import get_settings

    settings = get_settings()
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

    # --- Export golden eval set (shape consumed by eval.run_eval) ---
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
