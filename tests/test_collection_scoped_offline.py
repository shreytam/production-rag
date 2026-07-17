from core.types import ACLContext, Chunk
from providers.sparse.bm25 import BM25Retriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


def _chunk(cid, col, text):
    return Chunk(chunk_id=cid, doc_id=cid, text=text, tenant_id="t", collection_id=col)


def test_bm25_collection_scoping():
    r = BM25Retriever()
    r.index([_chunk("a", "A", "alpha shared"), _chunk("b", "B", "beta shared")])
    acl = ACLContext(tenant_id="t")
    got = r.search("shared", 5, acl, collection_id="A")
    assert [s.chunk.chunk_id for s in got] == ["a"]
    assert {s.chunk.chunk_id for s in r.search("shared", 5, acl)} == {"a", "b"}


def test_inmemory_vector_collection_scoping():
    emb = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [_chunk("a", "A", "alpha shared"), _chunk("b", "B", "beta shared")]
    for c, v in zip(chunks, emb.embed_documents([c.text for c in chunks])):
        c.embedding = v
    store.upsert(chunks)
    acl = ACLContext(tenant_id="t")
    got = store.search(emb.embed_query("shared"), 5, acl, collection_id="A")
    assert [s.chunk.chunk_id for s in got] == ["a"]
