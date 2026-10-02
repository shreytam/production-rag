from core.config import Settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Answer, Chunk, Query, ScoredChunk, Usage
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


class SpyReranker:
    def __init__(self):
        self.query = None

    def rerank(self, query, chunks, top_n):
        self.query = query
        return chunks[:top_n]


def _hybrid(reranker):
    emb, store, sparse = FakeEmbedder(), InMemoryVectorStore(), BM25Retriever()
    c = Chunk(chunk_id="a", doc_id="a", text="alpha beta", tenant_id="t")
    c.embedding = emb.embed_documents([c.text])[0]
    store.upsert([c])
    sparse.index([c])
    return HybridRetriever(emb, store, sparse, reranker)


def test_reranker_scores_against_original_question_when_provided():
    rr = SpyReranker()
    _hybrid(rr).retrieve(
        Query(text="alpha EXPANDED", rerank_text="alpha", acl=ACLContext(tenant_id="t"))
    )
    assert rr.query == "alpha"


def test_reranker_falls_back_to_query_text_without_rerank_text():
    rr = SpyReranker()
    _hybrid(rr).retrieve(Query(text="alpha", acl=ACLContext(tenant_id="t")))
    assert rr.query == "alpha"


def test_pipeline_threads_original_question_as_rerank_text():
    seen = {}

    class Rw:
        def rewrite(self, q, acl):
            return q + " EXPANDED"

    class Ret:
        def retrieve(self, q):
            seen["q"] = q
            return [ScoredChunk(chunk=Chunk(chunk_id="c", doc_id="d", text="x", tenant_id="t"), score=1.0)]

    class Gen:
        def generate(self, question, scored):
            return Answer(text="ok", contexts=list(scored), usage=Usage())

    p = RAGPipeline(Ret(), Gen(), Settings(), guardrails=None, rewriter=Rw())
    p.answer("original", ACLContext(tenant_id="t"))
    assert seen["q"].text == "original EXPANDED"
    assert seen["q"].rerank_text == "original"
