import pytest
from core.pipeline import build, HybridIndexError
from core.config import Settings

def test_pipeline_raises_on_empty_sparse_require_true(tmp_path):
    settings = Settings(
        sparse_index_dir=str(tmp_path),
        hybrid_require_sparse=True,
        nvidia_api_key="mock-key",
    )
    with pytest.raises(HybridIndexError):
        build(version="full", corpus="missing", settings=settings)

def test_pipeline_warning_on_empty_sparse_require_false(tmp_path):
    settings = Settings(
        sparse_index_dir=str(tmp_path),
        hybrid_require_sparse=False,
        nvidia_api_key="mock-key",
    )
    pipeline = build(version="full", corpus="missing", settings=settings)
    assert pipeline is not None
