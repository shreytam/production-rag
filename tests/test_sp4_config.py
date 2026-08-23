from core.config import Settings
from core.interfaces import SparseIndexLoader

def test_sp4_config_fields():
    settings = Settings(
        hybrid_require_sparse=True,
        sparse_index_dir=".cache",
        context_tokenizer="auto",
        context_token_safety_margin=0.10,
        chunk_overlap=32
    )
    assert settings.hybrid_require_sparse is True
    assert settings.context_tokenizer == "auto"

def test_sparse_index_loader_protocol():
    assert isinstance(SparseIndexLoader, type)
