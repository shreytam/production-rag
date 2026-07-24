from core.config import Settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Answer, Chunk, ScoredChunk, Usage


class SpyRewriter:
    def __init__(self):
        self.seen = []

    def rewrite(self, query, acl):
        self.seen.append(query)
        return query + " EXPANDED"


class SpyRetriever:
    def __init__(self):
        self.query_text = None

    def retrieve(self, q):
        self.query_text = q.text
        return [
            ScoredChunk(
                chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx", tenant_id="t1"),
                score=1.0,
            )
        ]


class SpyGrounded:
    def __init__(self):
        self.gen_question = None

    def generate(self, question, scored):
        self.gen_question = question
        return Answer(text="ok", contexts=list(scored), usage=Usage())


def test_rewriter_feeds_retrieval_but_not_generation():
    rw, ret, gen = SpyRewriter(), SpyRetriever(), SpyGrounded()
    p = RAGPipeline(ret, gen, Settings(), guardrails=None, embedder=None, rewriter=rw)
    p.answer("original question here", ACLContext(tenant_id="t1"))
    assert rw.seen == ["original question here"]
    assert ret.query_text == "original question here EXPANDED"  # retrieval uses rewrite
    assert gen.gen_question == "original question here"  # generation uses original


def test_no_rewriter_is_passthrough():
    ret, gen = SpyRetriever(), SpyGrounded()
    p = RAGPipeline(ret, gen, Settings(), guardrails=None, embedder=None, rewriter=None)
    p.answer("plain question", ACLContext(tenant_id="t1"))
    assert ret.query_text == "plain question"
    assert gen.gen_question == "plain question"
