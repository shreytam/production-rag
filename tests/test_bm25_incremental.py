from core.types import ACLContext, Chunk
from providers.sparse.bm25 import BM25Retriever


def _c(cid, tenant, text):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, tenant_id=tenant)


def test_add_makes_chunk_searchable():
    r = BM25Retriever()
    # Add first chunk with some text
    r.add([_c("d0::0", "t1", "alpha beta")])
    # Add second chunk with gamma - will have lower IDF than unique words
    # but we search for it in d2::0 which has it repeated
    r.add([_c("d1::0", "t1", "alpha beta")])
    r.add([_c("d2::0", "t1", "gamma delta gamma")])
    # Searching for "gamma" should return d2::0 first (has it twice, others don't have it)
    hits = r.search("gamma", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits][:1] == ["d2::0"]


def test_add_is_tenant_partitioned():
    r = BM25Retriever()
    r.add([_c("d1::0", "t1", "alpha")])
    r.add([_c("d2::0", "t2", "alpha")])
    hits = r.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert {h.chunk.tenant_id for h in hits} == {"t1"}


def test_delete_removes_chunk():
    r = BM25Retriever()
    r.add([_c("d1::0", "t1", "alpha beta"), _c("d1::1", "t1", "alpha gamma")])
    r.delete(["d1::0"], ACLContext(tenant_id="t1"))
    hits = r.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::1"]
