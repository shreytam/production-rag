"""Central configuration — the single source of every swappable knob.

One env var flips a provider; nothing else in the codebase reads env directly.
Defaults target NVIDIA NIM (OpenAI-compatible at https://integrate.api.nvidia.com/v1)
so the whole stack runs on one `nvapi-` key. Point EMBED_*/GEN_* at api.openai.com
to switch to real OpenAI without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class Settings(BaseSettings):
    # Load from the repo-root .env first, then infra/.env (where .env.example lives,
    # a natural place to drop the key). Later files override earlier ones, so a
    # root .env wins if both exist.
    model_config = SettingsConfigDict(
        env_file=("infra/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Shared keys (used as fallback when a component key is unset) ---
    nvidia_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Vector store: one switch ---
    vector_store: Literal["qdrant", "pgvector"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    pg_table: str = "rag_chunks"

    # --- Embeddings (OpenAI-compatible; dimension drives store schema) ---
    embed_base_url: str = NIM_BASE_URL
    embed_model: str = "baai/bge-m3"  # multilingual, 1024-d, long-context retrieval
    embed_dimension: int = 1024
    embed_api_key: str = ""
    embed_batch_size: int = 64

    # --- Generation (answer synthesis) ---
    gen_provider: Literal["openai", "anthropic"] = "openai"
    gen_base_url: str = NIM_BASE_URL
    gen_model: str = "meta/llama-3.3-70b-instruct"
    gen_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Cheap model for the contextual-prefix step (one call per chunk) ---
    context_provider: Literal["openai", "anthropic"] = "openai"
    context_base_url: str = NIM_BASE_URL
    context_model: str = "meta/llama-3.1-8b-instruct"
    context_api_key: str = ""
    anthropic_context_model: str = "claude-haiku-4-5-20251001"

    # --- LLM judge / RAGAS backing model ---
    judge_provider: Literal["openai", "anthropic"] = "openai"
    judge_base_url: str = NIM_BASE_URL
    judge_model: str = "meta/llama-3.3-70b-instruct"
    judge_api_key: str = ""
    anthropic_judge_model: str = "claude-sonnet-4-6"

    # --- Reranker: one switch ---
    reranker: Literal["local", "nim"] = "local"
    reranker_local_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_nim_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    reranker_nim_base_url: str = NIM_BASE_URL
    reranker_nim_api_key: str = ""

    # --- HTTP resilience (NIM can be slow under high traffic) ---
    # Generous per-request ceiling + several automatic retries with the OpenAI
    # SDK's built-in exponential backoff. Kept >= the SDK's 600s default so a
    # slow-but-valid response is awaited rather than aborted.
    request_timeout_seconds: float = 600.0
    max_retries: int = 5

    # --- Retrieval / assembly knobs ---
    rrf_k: int = 60
    retrieve_top_k: int = 20
    rerank_top_n: int = 8
    context_token_budget: int = 4000

    # --- Guardrails ---
    # On for the production query path (api/demo). The eval path forces this OFF
    # explicitly (CitationGuardrail/SchemaGuardrail would BLOCK normal answers and
    # confound metrics; groundedness adds an LLM call per item).
    guardrails_enabled: bool = True

    # --- Observability ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- Ingest sizing (keep corpora small to respect NIM rate limits) ---
    max_chunks_per_corpus: int = 2000
    contextual_cache_dir: str = ".cache/contextual"

    @model_validator(mode="after")
    def _fill_key_fallbacks(self) -> "Settings":
        """Component keys fall back to the shared NVIDIA key (or OpenAI when the
        base url points at OpenAI)."""
        def pick(explicit: str, base_url: str) -> str:
            if explicit:
                return explicit
            if "openai.com" in base_url:
                return self.openai_api_key or self.nvidia_api_key
            return self.nvidia_api_key or self.openai_api_key

        self.embed_api_key = pick(self.embed_api_key, self.embed_base_url)
        self.gen_api_key = pick(self.gen_api_key, self.gen_base_url)
        self.context_api_key = pick(self.context_api_key, self.context_base_url)
        self.judge_api_key = pick(self.judge_api_key, self.judge_base_url)
        self.reranker_nim_api_key = pick(self.reranker_nim_api_key, self.reranker_nim_base_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
