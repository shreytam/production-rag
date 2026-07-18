from core.types import ACLContext, Chunk, Query
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore


def _c(cid, col, text):
    return Chunk(chunk_id=cid, doc_id=cid, text=text, tenant_id="t", collection_id=col)


def test_retrieve_scopes_to_collection():
    emb, store, sparse = FakeEmbedder(), InMemoryVectorStore(), BM25Retriever()
    chunks = [_c("a", "A", "shared alpha"), _c("b", "B", "shared beta")]
    for c, v in zip(chunks, emb.embed_documents([c.text for c in chunks])):
        c.embedding = v
    store.upsert(chunks)
    sparse.index(chunks)
    r = HybridRetriever(emb, store, sparse, FakeReranker())
    hits = r.retrieve(Query(text="shared", acl=ACLContext(tenant_id="t"),
                            top_k=5, rerank_top_n=3, collection_id="A"))
    assert {h.chunk.chunk_id for h in hits} == {"a"}
