import sys

import cache.semantic_cache as sc


def test_importing_cache_does_not_import_redisvl():
    # Neither the seam nor the backend module may pull redis-vl at import time.
    import cache._redisvl_backend  # noqa: F401
    assert "redisvl" not in sys.modules
    assert "redis_vl" not in sys.modules


def test_build_cache_constructs_two_named_tiers_without_connecting():
    from core.config import Settings
    answer, retrieval = sc.build_cache(Settings())
    assert answer.index_name == "rag_cache_answer"
    assert retrieval.index_name == "rag_cache_retrieval"
    # Construction must not require Redis or redis-vl to be installed/reachable.
