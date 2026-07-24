from core.config import Settings


def test_sp12_rewriter_defaults():
    s = Settings()
    assert s.rewriter_enabled is True
    assert s.rewriter_llm_enabled is True
    assert s.rewriter_llm_threshold == 5
    # Reuses the existing shared Redis URL — no dedicated rewriter url knob.
    assert not hasattr(s, "rewriter_redis_url")
    assert s.redis_url.startswith("redis://")
