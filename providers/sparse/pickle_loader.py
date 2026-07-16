import pickle
from pathlib import Path
from core.config import Settings
from core.interfaces import SparseRetriever
from providers.sparse.bm25 import BM25Retriever

class PickleSparseIndexLoader:
    """Loads BM25Retriever from pickled files on disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self, corpus: str, store: str) -> SparseRetriever | None:
        if not corpus or not store:
            return None

        index_dir = Path(self.settings.sparse_index_dir)
        filename = f"bm25_{corpus}_{store}.pkl"
        filepath = index_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, "rb") as f:
                retriever = pickle.load(f)

            # D11 checks: check it unpickles to BM25Retriever containing _indices dict with len(_indices) >= 1
            if not isinstance(retriever, BM25Retriever):
                return None
            if not hasattr(retriever, "_indices") or not isinstance(retriever._indices, dict):
                return None
            if len(retriever._indices) < 1:
                return None

            return retriever
        except Exception:
            return None
