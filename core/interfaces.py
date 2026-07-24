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
    DocManifest,
    DocumentRecord,
    DocumentStatus,
    GuardrailResult,
    LLMResponse,
    Principal,
    Query,
    ChatMessage,
    ScoredChunk,
    Vector,
    PIISpan,
)


class AuthError(Exception):
    """Raised by an AuthVerifier when a token cannot be trusted.

    `status` is the HTTP status the API layer should surface: 401 (unauthenticated)
    or 403 (authenticated but no valid tenant / over-scoped token).
    """

    def __init__(self, detail: str, status: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


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

    def search(self, embedding: Vector, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]: ...

    def count(self, acl: ACLContext | None = None) -> int: ...

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None: ...

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None: ...


@runtime_checkable
class SparseRetriever(Protocol):
    """BM25/lexical store. ACL restricts the candidate set BEFORE scoring."""

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, top_k: int, acl: ACLContext, *,
               collection_id: str | None = None) -> list[ScoredChunk]: ...

    def add(self, chunks: list[Chunk]) -> None: ...

    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """Hybrid retriever: dense.search(acl) + sparse.search(acl) -> RRF -> rerank."""

    def retrieve(self, query: Query) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranking of candidate chunks against the query."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrites/expands a query for retrieval. MUST be fail-soft: any internal
    error returns a best-effort query rather than raising into the query path."""

    def rewrite(self, query: str, acl: ACLContext) -> str: ...


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
        seed: int | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class Guardrail(Protocol):
    """Input or output safety check. Treat all retrieved/user text as untrusted."""

    @property
    def name(self) -> str: ...

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult: ...


@runtime_checkable
class AuthVerifier(Protocol):
    """Turns a bearer token into a verified Principal. Raises AuthError on ANY
    failure — implementations must fail closed."""

    def verify(self, token: str) -> Principal: ...


@runtime_checkable
class TenantAllowlist(Protocol):
    """Per-tenant permitted acl_tags. `allowed()` returns None when unrestricted
    (claims pass through), else the frozenset of tags the tenant may hold."""

    def allowed(self, tenant_id: str) -> frozenset[str] | None: ...


@runtime_checkable
class PIIDetector(Protocol):
    """Contract for a PII detection engine."""
    def detect(self, text: str) -> list[PIISpan]: ...


@runtime_checkable
class SparseIndexLoader(Protocol):
    """Loads a persisted sparse index from disk."""

    def load(self, corpus: str, store: str) -> SparseRetriever | None: ...


@runtime_checkable
class ManifestStore(Protocol):
    def load(self, tenant_id: str, doc_id: str) -> "DocManifest | None": ...
    def save(self, manifest: "DocManifest") -> None: ...
    def delete(self, tenant_id: str, doc_id: str) -> None: ...


@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


@runtime_checkable
class DocumentRegistry(Protocol):
    def create(self, record: "DocumentRecord") -> None: ...
    def get(self, document_id: str, tenant_id: str) -> "DocumentRecord | None": ...
    def list(self, tenant_id: str) -> "list[DocumentRecord]": ...
    def set_status(self, document_id: str, tenant_id: str, status: "DocumentStatus",
                   *, error: str = "", chunk_count: int = 0) -> None: ...

