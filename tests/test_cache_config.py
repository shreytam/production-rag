from core.config import Settings


def test_cache_defaults_are_conservative():
    s = Settings()
    assert s.cache_enabled is False
    assert s.cache_similarity_threshold == 0.9
    assert s.cache_ttl_seconds == 3600
