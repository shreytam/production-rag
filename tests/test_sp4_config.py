import pytest
from pydantic import ValidationError

from core.config import Settings
from core.interfaces import SparseIndexLoader

def test_sp4_config_fields():
    settings = Settings(
        active_corpus="hotpotqa",
        hybrid_require_sparse=True,
        sparse_index_dir=".cache",
        context_tokenizer="auto",
        context_token_safety_margin=0.10,
        chunk_overlap=32
    )
    assert settings.active_corpus == "hotpotqa"
    assert settings.hybrid_require_sparse is True
    assert settings.context_tokenizer == "auto"
    assert settings.chunk_overlap == 32

def test_sparse_index_loader_protocol():
    assert isinstance(SparseIndexLoader, type)

def test_chunk_config_defaults_are_live():
    # P0 fix: chunk_overlap's declared default of 200 now actually takes
    # effect on both ingest paths (it used to be silently ignored). chunk_max_tokens
    # is a new knob for the chunker's token budget, also threaded through.
    settings = Settings()
    assert settings.chunk_max_tokens == 256
    assert settings.chunk_overlap == 200

def test_chunk_overlap_must_be_less_than_max_tokens():
    # An overlap >= max_tokens cannot produce forward progress while chunking
    # (see ingest/chunking.py's step = max_tokens - overlap) and must not boot.
    with pytest.raises(ValidationError):
        Settings(chunk_max_tokens=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        Settings(chunk_max_tokens=100, chunk_overlap=150)
    # Equal to the boundary below max_tokens is legal.
    settings = Settings(chunk_max_tokens=100, chunk_overlap=99)
    assert settings.chunk_overlap == 99
