from cache.semantic_cache import acl_key
from tests.cache.fake_cache import FakeSemanticCache

EMB = [1.0, 0.0]


def _store(c, tags, payload, doc_ids=("d1",)):
    c.store(tenant_id="acme", collection_id=None, acl_tags=tags, embedding=EMB,
            payload=payload, doc_ids=list(doc_ids))


def _look(c, tags):
    return c.lookup(tenant_id="acme", collection_id=None, acl_tags=tags, embedding=EMB)


def test_untagged_user_cannot_read_tagged_entry():
    c = FakeSemanticCache()
    _store(c, ("hr",), {"v": "secret"})
    assert _look(c, ()) is None
    assert _look(c, ("hr",)) == {"v": "secret"}
    assert _look(c, ("hr", "finance")) is None


def test_tag_order_and_duplicates_irrelevant():
    c = FakeSemanticCache()
    _store(c, ("b", "a", "a"), {"v": 1})
    assert _look(c, ("a", "b")) == {"v": 1}


def test_invalidation_evicts_across_acl_partitions():
    c = FakeSemanticCache()
    _store(c, ("hr",), {"v": 1})
    _store(c, (), {"v": 2})
    assert c.invalidate_document(tenant_id="acme", collection_id=None, doc_id="d1") == 2
    assert _look(c, ("hr",)) is None and _look(c, ()) is None


def test_acl_key_canonical_and_distinct_for_empty():
    assert acl_key(("b", "a", "a")) == acl_key(["a", "b"])
    assert acl_key(()) != acl_key(("",))
    assert acl_key(()) != acl_key(("a",))
