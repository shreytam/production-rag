from tests.cache.fake_cache import FakeSemanticCache


def _clock():
    box = {"t": 1000.0}
    return box, (lambda: box["t"])


def test_hit_within_threshold_miss_outside():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    # identical vector -> hit
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) == {"v": 1}
    # orthogonal vector (cosine 0) -> miss
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0]) is None


def test_never_crosses_tenant():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    assert c.lookup(tenant_id="other", collection_id=None, embedding=[1.0, 0.0]) is None


def test_never_crosses_collection():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id="kb", embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    # unscoped query must not read the collection-scoped entry
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None


def test_targeted_eviction_removes_only_referencing_entries():
    c = FakeSemanticCache(threshold=0.9)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1", "d2"])
    c.store(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0],
            payload={"v": 2}, doc_ids=["d3"])
    n = c.invalidate_document(tenant_id="acme", collection_id=None, doc_id="d1")
    assert n == 1
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[0.0, 1.0]) == {"v": 2}


def test_ttl_expiry_via_injected_clock():
    box, now = _clock()
    c = FakeSemanticCache(threshold=0.9, ttl_seconds=100, time_fn=now)
    c.store(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0],
            payload={"v": 1}, doc_ids=["d1"])
    box["t"] = 1050.0  # within TTL
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) == {"v": 1}
    box["t"] = 1101.0  # past TTL
    assert c.lookup(tenant_id="acme", collection_id=None, embedding=[1.0, 0.0]) is None
