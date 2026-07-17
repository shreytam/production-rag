# tests/test_acl_collection_filter.py
from core.types import ACLContext, Chunk
from retrieval.acl import acl_predicate, qdrant_filter


def _chunk(collection_id):
    return Chunk(chunk_id="c", doc_id="d", text="x", tenant_id="t", collection_id=collection_id)


def test_acl_predicate_collection_filter():
    acl = ACLContext(tenant_id="t")
    pred = acl_predicate(acl, collection_id="A")
    assert pred(_chunk("A")) is True
    assert pred(_chunk("B")) is False
    # No collection filter => collection ignored
    assert acl_predicate(acl)(_chunk("B")) is True


def test_qdrant_filter_appends_collection():
    f = qdrant_filter(ACLContext(tenant_id="t"), collection_id="A")
    keys = [getattr(c, "key", None) for c in f.must]
    assert "collection_id" in keys
