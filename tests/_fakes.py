"""Shared offline fakes for security/integration tests (not collected as tests)."""

from __future__ import annotations

import math

from core.types import (
    ACLContext,
    ChatMessage,
    Chunk,
    LLMResponse,
    RetrievalSource,
    ScoredChunk,
    Usage,
)

DIM = 16


def vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in text.lower().split():
        v[hash(tok) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    dimension = DIM

    def embed_documents(self, texts):
        return [vec(t) for t in texts]

    def embed_query(self, text):
        return vec(text)


class InMemoryVectorStore:
    """ACL applied BEFORE scoring, mirroring the real stores' pre-similarity filter."""

    def __init__(self):
        self.chunks: list[Chunk] = []

    def ensure_collection(self, dimension):
        pass

    def upsert(self, chunks):
        self.chunks.extend(chunks)

    def search(self, embedding, top_k, acl: ACLContext):
        scored = []
        for c in self.chunks:
            if not acl.allows(c.acl):  # pre-similarity ACL gate
                continue
            sim = sum(a * b for a, b in zip(embedding, c.embedding or []))
            scored.append(ScoredChunk(chunk=c, score=sim, source=RetrievalSource.DENSE))
        scored.sort(key=lambda s: s.score, reverse=True)
        for r, s in enumerate(scored[:top_k], 1):
            s.rank = r
        return scored[:top_k]

    def count(self, acl=None):
        return len(self.chunks)

    def delete(self, chunk_ids, acl):
        wanted = set(chunk_ids)
        self.chunks = [
            c for c in self.chunks
            if not (c.chunk_id in wanted and c.tenant_id == acl.tenant_id)
        ]

    def update_metadata(self, updates, acl):
        for c in self.chunks:
            if c.tenant_id != acl.tenant_id:
                continue
            payload = updates.get(c.chunk_id)
            if payload and "title" in payload:
                c.title = payload["title"]


class FakeReranker:
    def rerank(self, query, chunks, top_n):
        qtok = set(query.lower().split())
        ranked = sorted(
            chunks,
            key=lambda sc: len(qtok & set(sc.chunk.text.lower().split())),
            reverse=True,
        )
        out = []
        for r, sc in enumerate(ranked[:top_n], 1):
            out.append(
                ScoredChunk(chunk=sc.chunk, score=float(top_n - r), source=RetrievalSource.RERANK, rank=r)
            )
        return out


class RecordingGenerator:
    """Generator that records the messages it receives and returns a canned reply.

    `parsed` is returned when a response_model is requested; otherwise `text`.
    """

    model = "fake"

    def __init__(self, text="OK", parsed=None):
        self._text = text
        self._parsed = parsed
        self.calls: list[list[ChatMessage]] = []
        self.last_seed = None

    def complete(self, messages, *, response_model=None, seed=None, **_):
        self.calls.append(list(messages))
        self.last_seed = seed
        return LLMResponse(
            text=self._text,
            parsed=self._parsed if response_model else None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake",
        )
