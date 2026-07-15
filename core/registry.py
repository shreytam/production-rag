"""Factory: build concrete components from Settings.

This is the ONLY module that names concrete implementation classes. Everything
else depends on the Protocols in `core.interfaces`. Swapping a provider is a
config change, resolved here. Imports are local so a missing (not-yet-built)
implementation only fails when that specific component is requested.

Contract for Wave 1 implementors — produce exactly these classes/constructors:
  Embedder:    providers.embedders.openai_compatible.OpenAICompatibleEmbedder(settings)
  VectorStore: providers.vectorstores.qdrant_store.QdrantVectorStore(settings)
               providers.vectorstores.pgvector_store.PgVectorStore(settings)
  Sparse:      providers.sparse.bm25.BM25Retriever()
  Reranker:    providers.rerankers.local_cross_encoder.LocalCrossEncoderReranker(model)
               providers.rerankers.nim_rerank.NIMReranker(model, base_url, api_key)
  Generator:   providers.generators.openai_compatible.OpenAICompatibleGenerator(model, base_url, api_key)
               providers.generators.anthropic.AnthropicGenerator(model, api_key)
"""

from __future__ import annotations

from typing import Literal

from core.config import Settings, get_settings
from core.interfaces import Embedder, Generator, Reranker, SparseRetriever, VectorStore, PIIDetector

GeneratorRole = Literal["gen", "context", "judge"]


def build_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    from providers.embedders.openai_compatible import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(s)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    s = settings or get_settings()
    if s.vector_store == "qdrant":
        from providers.vectorstores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(s)
    if s.vector_store == "pgvector":
        from providers.vectorstores.pgvector_store import PgVectorStore

        return PgVectorStore(s)
    raise ValueError(f"Unknown vector_store: {s.vector_store}")


def build_sparse_retriever(settings: Settings | None = None) -> SparseRetriever:
    from providers.sparse.bm25 import BM25Retriever

    return BM25Retriever()


def build_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    if s.reranker == "local":
        from providers.rerankers.local_cross_encoder import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(s.reranker_local_model)
    if s.reranker == "nim":
        from providers.rerankers.nim_rerank import NIMReranker

        return NIMReranker(
            s.reranker_nim_model, s.reranker_nim_base_url, s.reranker_nim_api_key
        )
    raise ValueError(f"Unknown reranker: {s.reranker}")


def build_generator(role: GeneratorRole = "gen", settings: Settings | None = None) -> Generator:
    """Build the generator for a role (answer gen / contextual prefix / judge).
    Each role has its own provider + model knobs so the cheap model handles the
    per-chunk contextual step."""
    s = settings or get_settings()
    provider = {"gen": s.gen_provider, "context": s.context_provider, "judge": s.judge_provider}[role]
    if provider == "openai":
        from providers.generators.openai_compatible import OpenAICompatibleGenerator

        model = {"gen": s.gen_model, "context": s.context_model, "judge": s.judge_model}[role]
        base_url = {"gen": s.gen_base_url, "context": s.context_base_url, "judge": s.judge_base_url}[role]
        api_key = {"gen": s.gen_api_key, "context": s.context_api_key, "judge": s.judge_api_key}[role]
        return OpenAICompatibleGenerator(
            model,
            base_url,
            api_key,
            timeout=s.request_timeout_seconds,
            max_retries=s.max_retries,
        )
    if provider == "anthropic":
        from providers.generators.anthropic import AnthropicGenerator

        model = {
            "gen": s.anthropic_model,
            "context": s.anthropic_context_model,
            "judge": s.anthropic_judge_model,
        }[role]
        return AnthropicGenerator(model, s.anthropic_api_key)
    raise ValueError(f"Unknown provider for role {role}: {provider}")


def build_auth_verifier(settings: Settings | None = None):
    """Build the JWT verifier from config (the only place its class is named)."""
    s = settings or get_settings()
    from providers.auth.jwt_verifier import JWTVerifier

    return JWTVerifier(
        alg=s.jwt_alg,
        hs_secret=s.jwt_secret,
        jwks_url=s.jwks_url,
        issuer=s.jwt_issuer,
        audience=s.jwt_audience,
        leeway_seconds=s.jwt_leeway_seconds,
        max_acl_tags=s.max_acl_tags,
    )


def build_allowlist(settings: Settings | None = None):
    """Build the tenant allowlist: NullAllowlist unless acl_allowlist_source is set."""
    s = settings or get_settings()
    if not s.acl_allowlist_source:
        from providers.auth.allowlist import NullAllowlist

        return NullAllowlist()
    from providers.auth.allowlist import StaticAllowlist

    return StaticAllowlist.from_file(s.acl_allowlist_source)


def build_pii_detector(settings: Settings | None = None) -> PIIDetector:
    s = settings or get_settings()
    if s.pii_detector == "regex":
        from providers.pii.regex_detector import RegexPIIDetector
        return RegexPIIDetector()
    elif s.pii_detector == "presidio":
        from providers.pii.presidio_detector import PresidioPIIDetector
        return PresidioPIIDetector()
    raise ValueError(f"Unknown pii_detector configured: {s.pii_detector}")

