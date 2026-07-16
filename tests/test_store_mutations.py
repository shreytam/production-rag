from core.types import ACLContext, Chunk
from tests._fakes import InMemoryVectorStore, vec


def _chunk(cid, tenant, text="hello world"):
    return Chunk(chunk_id=cid, doc_id="d1", text=text, tenant_id=tenant,
                 embedding=vec(text))


def test_delete_is_tenant_scoped():
    store = InMemoryVectorStore()
    store.upsert([_chunk("c1", "t1"), _chunk("c2", "t2")])
    store.delete(["c1", "c2"], ACLContext(tenant_id="t1"))
    remaining = {c.chunk_id for c in store.chunks}
    assert remaining == {"c2"}  # c2 belongs to t2, untouched by a t1 caller


def test_update_metadata_is_tenant_scoped():
    store = InMemoryVectorStore()
    store.upsert([_chunk("c1", "t1"), _chunk("c2", "t2")])
    store.update_metadata({"c1": {"title": "X"}, "c2": {"title": "Y"}},
                          ACLContext(tenant_id="t1"))
    by_id = {c.chunk_id: c for c in store.chunks}
    assert by_id["c1"].title == "X"
    assert by_id["c2"].title != "Y"  # cross-tenant update refused
