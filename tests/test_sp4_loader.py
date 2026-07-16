import pytest
from pathlib import Path
from core.registry import build_sparse_retriever
from providers.sparse.pickle_loader import PickleSparseIndexLoader
from providers.sparse.bm25 import BM25Retriever
from core.config import Settings

def test_pickle_loader_not_found(tmp_path):
    settings = Settings(sparse_index_dir=str(tmp_path))
    loader = PickleSparseIndexLoader(settings)
    assert loader.load("missing_corpus", "qdrant") is None

def test_build_sparse_retriever_falls_back_on_miss(tmp_path):
    settings = Settings(sparse_index_dir=str(tmp_path))
    retriever = build_sparse_retriever(settings, corpus="missing")
    assert isinstance(retriever, BM25Retriever)
    assert len(retriever._indices) == 0
