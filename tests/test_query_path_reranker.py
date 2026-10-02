import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.types import ACLContext, Chunk, Query, RetrievalSource, ScoredChunk
from providers.rerankers.nim_rerank import NIMReranker
from providers.rerankers.openrouter_rerank import OpenRouterReranker
from providers.sparse.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from tests._fakes import FakeEmbedder, InMemoryVectorStore


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)


def _chunks():
    return [
        ScoredChunk(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", tenant_id="t"),
                    score=0.1, source=RetrievalSource.DENSE)
        for i in range(2)
    ]


def _status_resp(code):
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(code, request=req)
    m = MagicMock()
    m.raise_for_status.side_effect = httpx.HTTPStatusError("e", request=req, response=resp)
    return m


def _ok(payload):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = payload
    return m


def _nim():
    return NIMReranker("m", "http://x", "k"), {"rankings": [{"index": 1, "logit": 0.9}]}


def _or():
    return OpenRouterReranker("m", "http://x", "k"), {"results": [{"index": 1, "relevance_score": 0.9}]}


@pytest.mark.parametrize("make", [_nim, _or])
@pytest.mark.parametrize("code", [429, 500, 503])
def test_retries_on_429_and_5xx(make, code):
    rr, payload = make()
    with patch("httpx.Client.post", side_effect=[_status_resp(code), _ok(payload)]) as post:
        out = rr.rerank("q", _chunks(), 1)
    assert post.call_count == 2
    assert out[0].chunk.chunk_id == "c1"


@pytest.mark.parametrize("make", [_nim, _or])
def test_does_not_retry_4xx_client_errors(make):
    rr, _ = make()
    with patch("httpx.Client.post", side_effect=[_status_resp(401), _status_resp(401)]) as post:
        with pytest.raises(httpx.HTTPStatusError):
            rr.rerank("q", _chunks(), 1)
    assert post.call_count == 1


@pytest.mark.parametrize("make", [_nim, _or])
def test_retry_attempts_are_bounded(make):
    rr, _ = make()
    with patch("httpx.Client.post", side_effect=lambda *a, **k: _status_resp(503)) as post:
        with pytest.raises(httpx.HTTPStatusError):
            rr.rerank("q", _chunks(), 1)
    assert post.call_count == 3


class _BoomReranker:
    def rerank(self, query, chunks, top_n):
        raise RuntimeError("down")


class _EmptyReranker:
    def rerank(self, query, chunks, top_n):
        return []


def _hybrid(reranker):
    emb, store, sparse = FakeEmbedder(), InMemoryVectorStore(), BM25Retriever()
    c = Chunk(chunk_id="a", doc_id="a", text="alpha beta", tenant_id="t")
    c.embedding = emb.embed_documents([c.text])[0]
    store.upsert([c])
    sparse.index([c])
    return HybridRetriever(emb, store, sparse, reranker)


@pytest.mark.parametrize("rr", [_BoomReranker(), _EmptyReranker()])
def test_reranker_fallback_is_flagged_and_logged(rr, caplog):
    q = Query(text="alpha", acl=ACLContext(tenant_id="t"))
    with caplog.at_level(logging.WARNING):
        out = _hybrid(rr).retrieve(q)
    assert [s.chunk_id for s in out] == ["a"]
    assert q.metadata["reranker_fallback"] is True
    assert any("reranker" in r.message.lower() for r in caplog.records)


def test_no_fallback_flag_on_success():
    class Ok:
        def rerank(self, query, chunks, top_n):
            return chunks[:top_n]

    q = Query(text="alpha", acl=ACLContext(tenant_id="t"))
    _hybrid(Ok()).retrieve(q)
    assert "reranker_fallback" not in q.metadata


def test_pipeline_surfaces_reranker_fallback_in_trace_output():
    from core.config import Settings
    from core.pipeline import RAGPipeline
    from core.types import Answer, Usage

    spans = []

    class _Span:
        def __init__(self, name):
            self.name = name

        def update(self, **kw):
            spans.append((self.name, kw))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Tracer:
        def span(self, name, **kw):
            return _Span(name)

    class Gen:
        def generate(self, question, scored):
            return Answer(text="ok", contexts=list(scored), usage=Usage())

    p = RAGPipeline(_hybrid(_BoomReranker()), Gen(), Settings(), tracer=_Tracer(), guardrails=None)
    p.answer("alpha", ACLContext(tenant_id="t"))
    out = [kw["output"] for name, kw in spans if name == "retrieval" and "output" in kw]
    assert out and out[0]["reranker_fallback"] is True
