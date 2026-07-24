import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CACHE_LIVE_SMOKE") != "1",
    reason="set CACHE_LIVE_SMOKE=1 with a running Redis 8 to exercise redis-vl",
)


def test_redisvl_store_lookup_invalidate_round_trip():
    from cache._redisvl_backend import RedisVLSemanticCache
    from core.config import Settings

    s = Settings(cache_enabled=True, embed_dimension=4, cache_similarity_threshold=0.9)
    c = RedisVLSemanticCache(index_name="rag_cache_smoke", settings=s)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0, 0.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    assert c.lookup(tenant_id="acme", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) == {"v": 1}
    assert c.lookup(tenant_id="other", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) is None
    assert c.invalidate_document(tenant_id="acme", collection_id=None, doc_id="d1") == 1
    assert c.lookup(tenant_id="acme", collection_id=None,
                    embedding=[1.0, 0.0, 0.0, 0.0]) is None
