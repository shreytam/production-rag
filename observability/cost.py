"""Token-based cost estimation for models used in the RAG pipeline.

Prices are ballpark estimates based on publicly available information at the time
of writing — confirm actual rates with the provider before billing customers.
NIM free-tier models are listed at $0.00 (subject to NVIDIA's usage terms).
"""

from __future__ import annotations

from core.types import Usage

# ($ per 1k input tokens, $ per 1k output tokens)
# ESTIMATES — confirm with provider before use in production billing.
PRICING: dict[str, tuple[float, float]] = {
    # NVIDIA NIM free-tier (OpenAI-compatible endpoint) — $0.00 on free tier
    "meta/llama-3.3-70b-instruct": (0.00, 0.00),   # NIM free tier
    "meta/llama-3.1-8b-instruct": (0.00, 0.00),     # NIM free tier
    "nvidia/nv-embedqa-e5-v5": (0.00, 0.00),         # NIM free tier (embed, no completion)
    # Anthropic models — estimate from public pricing page
    "claude-sonnet-4-6": (0.003, 0.015),             # ~$3/Mtok in, ~$15/Mtok out (estimate)
    "claude-haiku-4-5-20251001": (0.00025, 0.00125), # ~$0.25/Mtok in, ~$1.25/Mtok out (estimate)
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in USD for a single LLM call.

    Returns 0.0 for unknown models — never raises.
    """
    if model not in PRICING:
        return 0.0
    input_rate, output_rate = PRICING[model]
    return (prompt_tokens / 1000.0) * input_rate + (completion_tokens / 1000.0) * output_rate


def cost_per_1k_queries(total_cost: float, n_queries: int) -> float:
    """Normalise total cost to per-1000-query rate.

    Returns 0.0 when n_queries is 0 to avoid ZeroDivisionError.
    """
    if n_queries == 0:
        return 0.0
    return (total_cost / n_queries) * 1000.0


def update_usage_cost(usage: Usage, model: str) -> Usage:
    """Return a copy of *usage* with ``cost_usd`` computed from token counts."""
    usd = cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    return usage.model_copy(update={"cost_usd": usd})
