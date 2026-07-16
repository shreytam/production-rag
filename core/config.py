"""Central configuration — the single source of every swappable knob.

One env var flips a provider; nothing else in the codebase reads env directly.
Defaults target NVIDIA NIM (OpenAI-compatible at https://integrate.api.nvidia.com/v1)
so the whole stack runs on one `nvapi-` key. Point EMBED_*/GEN_* at api.openai.com
to switch to real OpenAI without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator, Field, AliasChoices
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

    # --- OpenAI-compatible model router (one base_url + key for all roles) ---
    # Any model role left at the NIM default url / empty key inherits these.
    # The reranker is intentionally excluded (no OpenAI-standard rerank endpoint).
    llm_base_url: str = ""
    llm_api_key: str = ""

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
    active_corpus: str = ""
    # NOTE: build(version="full") now always uses the per-tenant TenantSparseStore,
    # so this flag currently has no effect on the pipeline (the fail-closed gate that
    # used to raise HybridIndexError on an empty/missing sparse index was removed).
    # Kept for config/backward compatibility only.
    hybrid_require_sparse: bool = True
    sparse_index_dir: str = ".cache"
    tenant_sparse_dir: str = ".cache/sparse_tenants"
    context_tokenizer: str = "auto"
    context_token_safety_margin: float = 0.0
    chunk_overlap: int = 200

    # --- Guardrails ---
    # On for the production query path (api/demo). The eval path forces this OFF
    # explicitly (CitationGuardrail/SchemaGuardrail would BLOCK normal answers and
    # confound metrics; groundedness adds an LLM call per item).
    guardrails_enabled: bool = True

    # SP2 guardrail-correctness knobs
    injection_llm_escalation: bool = True       # borderline input → 1 LLM classifier call
    groundedness_timeout_seconds: float = 20.0  # wall-clock bound on the faithfulness call

    # --- Auth & tenancy (SP1) ---
    app_env: Literal["dev", "prod"] = "dev"
    jwt_alg: Literal["HS256", "RS256"] = "HS256"
    jwt_secret: str = ""          # HS256 shared secret (dev); presence enables minting
    jwks_url: str = ""            # RS256 JWKS endpoint (prod)
    jwt_issuer: str = ""          # enforced on prod
    jwt_audience: str = ""        # enforced on prod
    jwt_leeway_seconds: int = 60
    acl_allowlist_source: str = ""  # empty = NullAllowlist; path = StaticAllowlist JSON
    auth_dev_signer_enabled: bool = False
    max_question_chars: int = 8000
    max_acl_tags: int = 32

    # --- Observability ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- PII & Compliance ---
    pii_mode: Literal["redact", "keep"] = "redact"
    pii_detector: Literal["regex", "presidio"] = "regex"
    pii_audit_log_path: str = ".audit/pii_audit.jsonl"
    pii_audit_value_hash: bool = False
    pii_audit_hash_salt: str | None = None
    pii_scan_output: bool = True
    langfuse_sample_rate: float = 1.0

    # --- Ingest sizing (keep corpora small to respect NIM rate limits) ---
    max_chunks_per_corpus: int = 2000
    contextual_cache_dir: str = ".cache/contextual"
    manifest_dir: str = ".cache/manifest"

    # --- Eval Gate & Stats ---
    eval_tolerance: float = 0.03
    eval_fast_n: int = 15
    eval_fast_seed: int = 0
    eval_bootstrap_resamples: int = 1000

    # --- LLM Judge Voting ---
    judge_votes: int = 3
    judge_seed: int = 0

    # --- Live Store CI Gating ---
    require_live_stores: bool = Field(
        default=False,
        validation_alias=AliasChoices("rag_require_live_stores", "require_live_stores")
    )

    @model_validator(mode="after")
    def _apply_llm_router(self) -> "Settings":
        """Point every model role at one OpenAI-compatible router unless the role
        was explicitly overridden. A role url still at the NIM default, or a role
        with an empty base url, adopts llm_base_url; empty role keys adopt
        llm_api_key. The reranker is deliberately not routed."""
        if self.llm_base_url:
            for field in ("embed_base_url", "gen_base_url", "context_base_url", "judge_base_url"):
                if getattr(self, field) in ("", NIM_BASE_URL):
                    setattr(self, field, self.llm_base_url)
        if self.llm_api_key:
            for field in ("embed_api_key", "gen_api_key", "context_api_key", "judge_api_key"):
                if not getattr(self, field):
                    setattr(self, field, self.llm_api_key)
        return self

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

    @model_validator(mode="after")
    def _validate_auth(self) -> "Settings":
        """Prod instances must be securely configured — fail fast at construction."""
        if self.app_env == "prod":
            if self.jwt_alg == "HS256" and not self.jwt_secret:
                raise ValueError("HS256 requires jwt_secret when app_env=prod")
            if self.jwt_alg == "RS256" and not self.jwks_url:
                raise ValueError("RS256 requires jwks_url when app_env=prod")
            if not self.jwt_issuer or not self.jwt_audience:
                raise ValueError("jwt_issuer and jwt_audience are required when app_env=prod")
            if self.auth_dev_signer_enabled:
                raise ValueError("auth_dev_signer_enabled must be False when app_env=prod")
        return self

    @model_validator(mode="after")
    def _validate_pii_settings(self) -> "Settings":
        if self.pii_audit_value_hash and not self.pii_audit_hash_salt:
            raise ValueError(
                "pii_audit_hash_salt must be set when pii_audit_value_hash is True to prevent hash brute-forcing."
            )
        return self



@lru_cache
def get_settings() -> Settings:
    return Settings()
