"""Component contracts for the RAG pipeline.

Everything is a `typing.Protocol` so concrete implementations stay framework-free
and swappable. The pipeline depends ONLY on these; `core.registry` is the single
place that names concrete classes. Wave 1 agents implement against this file —
do not change a signature without updating the registry and every implementor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from core.types import (
    ACLContext,
    Chunk,
    GuardrailResult,
    LLMResponse,
    Query,
    ChatMessage,
    ScoredChunk,
    Vector,
)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Dimension is config-driven, never hardcoded."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense store. ACL is a REQUIRED search argument applied pre-similarity."""

    def ensure_collection(self, dimension: int) -> None: ...

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, embedding: Vector, top_k: int, acl: ACLContext) -> list[ScoredChunk]: ...

    def count(self, acl: ACLContext | None = None) -> int: ...


@runtime_checkable
class SparseRetriever(Protocol):
    """BM25/lexical store. ACL restricts the candidate set BEFORE scoring."""

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, top_k: int, acl: ACLContext) -> list[ScoredChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Hybrid retriever: dense.search(acl) + sparse.search(acl) -> RRF -> rerank."""

    def retrieve(self, query: Query) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranking of candidate chunks against the query."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """LLM text generation. When `response_model` is given, the implementation
    must coerce output into that schema and populate `LLMResponse.parsed`."""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


@runtime_checkable
class Guardrail(Protocol):
    """Input or output safety check. Treat all retrieved/user text as untrusted."""

    @property
    def name(self) -> str: ...

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult: ...
