"""Shared data models for the RAG pipeline.

These are the wire types every component speaks. They are deliberately plain:
Pydantic models for validation + JSON serialization (eval artifacts, structured
LLM output), nothing framework-specific. Frozen as part of Wave 0 — treat changes
here as contract changes that ripple across every workstream.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Vector = list[float]


class ACLContext(BaseModel):
    """The authorization scope of a request OR the ownership tags of a chunk.

    SECURITY: at query time this is derived from the authenticated caller, never
    from the prompt or any retrieved document text. Stores must apply it as a
    pre-similarity filter.
    """

    tenant_id: str
    acl_tags: tuple[str, ...] = ()

    @field_validator("tenant_id")
    @classmethod
    def _acl_tenant_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ACLContext.tenant_id must be non-empty")
        return v

    def allows(self, owner: "ACLContext") -> bool:
        """True if a caller with THIS scope may see a chunk owned by `owner`.

        Tenant must match exactly. A chunk with no acl_tags is visible to anyone
        in the tenant; otherwise the caller must hold at least one matching tag.
        """
        if self.tenant_id != owner.tenant_id:
            return False
        if not owner.acl_tags:
            return True
        return bool(set(self.acl_tags) & set(owner.acl_tags))


class Principal(BaseModel):
    """A cryptographically verified caller identity. Built ONLY from a verified
    token's claims — never from raw request headers/body. `ACLContext` is derived
    from this, closing the tenant-spoofing hole."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    acl_tags: tuple[str, ...] = ()
    subject: str = ""
    claims: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def _tenant_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Principal.tenant_id must be non-empty")
        return v

    def to_acl(self) -> "ACLContext":
        return ACLContext(tenant_id=self.tenant_id, acl_tags=self.acl_tags)


class Document(BaseModel):
    """A source document prior to chunking."""

    doc_id: str
    text: str
    tenant_id: str
    acl_tags: tuple[str, ...] = ()
    collection_id: str = ""
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An indexed unit of retrieval."""

    chunk_id: str
    doc_id: str
    text: str
    tenant_id: str
    acl_tags: tuple[str, ...] = ()
    collection_id: str = ""
    ordinal: int = 0
    title: str | None = None
    source: str | None = None
    contextual_prefix: str | None = None
    embedding: Vector | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def acl(self) -> ACLContext:
        return ACLContext(tenant_id=self.tenant_id, acl_tags=self.acl_tags)

    @property
    def embed_text(self) -> str:
        """Text that gets embedded/indexed: contextual prefix prepended if present
        (Anthropic contextual-retrieval technique)."""
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.text}"
        return self.text


class RetrievalSource(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANK = "rerank"


class ScoredChunk(BaseModel):
    """A chunk with a relevance score and provenance."""

    chunk: Chunk
    score: float
    source: RetrievalSource = RetrievalSource.DENSE
    rank: int | None = None
    component_scores: dict[str, float] = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Query(BaseModel):
    """A retrieval request. `acl` is mandatory — there is no unscoped query path."""

    text: str
    acl: ACLContext
    top_k: int = 20
    rerank_top_n: int = 8
    collection_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class LLMResponse(BaseModel):
    """Raw return from a Generator.complete() call."""

    text: str
    parsed: dict[str, Any] | None = None  # present when a response schema was requested
    usage: Usage = Field(default_factory=Usage)
    model: str = ""


class Citation(BaseModel):
    """An inline citation tying a claim to a retrieved chunk."""

    marker: str  # e.g. "[1]"
    chunk_id: str
    doc_id: str = ""
    quote: str = ""


class Answer(BaseModel):
    """The grounded, cited response. This is also the structured-output schema the
    Generator is asked to fill (citations enforced downstream)."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    grounded: bool | None = None
    refused: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailAction(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    REDACT = "redact"


class GuardrailResult(BaseModel):
    """Outcome of a single guardrail check (input or output)."""

    name: str
    action: GuardrailAction = GuardrailAction.PASS
    reason: str = ""
    payload: str | None = None  # redacted text when action == REDACT
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.action != GuardrailAction.BLOCK


class PIISpan(BaseModel):
    """Represents a validated block of detected PII.

    SECURITY: Contains only coordinates and type; never holds the segment's raw text.
    """
    type: str
    start: int
    end: int

    model_config = {
        "frozen": True,
        "extra": "forbid"
    }


class ChunkRecord(BaseModel):
    chunk_id: str
    ordinal: int
    embed_hash: str
    meta_hash: str


class DocManifest(BaseModel):
    tenant_id: str
    doc_id: str
    prompt_version: str = "v1"
    chunks: dict[str, ChunkRecord] = Field(default_factory=dict)


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class DocumentRecord(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    blob_key: str
    collection_id: str = ""
    error: str = ""
    chunk_count: int = 0

