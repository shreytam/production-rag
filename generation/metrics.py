"""Faithfulness metric, moved verbatim from eval/generation_metrics.py when the benchmark/eval harness was removed."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.interfaces import Generator
from core.types import ChatMessage


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


class ClaimList(BaseModel):
    """Atomic claims extracted from an answer."""

    claims: list[str] = Field(description="Atomic factual claims extracted from the answer.")


class ClaimVerdict(BaseModel):
    """Verdict for a single claim against the provided contexts."""

    claim: str
    supported: bool = Field(description="True if the claim is supported by the contexts.")


class ClaimVerdicts(BaseModel):
    """Verdicts for all claims."""

    verdicts: list[ClaimVerdict]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def _parsed(response_parsed: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if response_parsed is None:
        return default
    return response_parsed.get(key, default)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    generator: Generator,
) -> float:
    """Fraction of atomic claims in *answer* that are supported by *contexts*.

    Faithfulness = supported_claims / total_claims.

    Steps
    -----
    1. Ask the LLM to extract atomic claims from *answer*.
    2. Ask the LLM to judge each claim as supported/not by *contexts*.
    """
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1 – claim extraction
    extract_resp = generator.complete(
        [
            _system("You are an expert at decomposing answers into atomic factual claims."),
            _user(
                f"Question: {question}\n\nAnswer: {answer}\n\n"
                "Extract every atomic factual claim made in the answer as a JSON list."
            ),
        ],
        response_model=ClaimList,
        max_tokens=512,
    )
    claims: list[str] = _parsed(extract_resp.parsed, "claims", [])
    if not claims:
        return 0.0

    # Step 2 – verdict per claim against contexts
    claims_block = "\n".join(f"- {c}" for c in claims)
    verdict_resp = generator.complete(
        [
            _system(
                "You are a factual verification expert. "
                "Determine whether each claim is supported by the provided contexts."
            ),
            _user(
                f"Contexts:\n{context_block}\n\nClaims:\n{claims_block}\n\n"
                "For each claim, output a verdict (supported: true/false)."
            ),
        ],
        response_model=ClaimVerdicts,
        max_tokens=512,
    )
    verdicts: list[dict] = _parsed(verdict_resp.parsed, "verdicts", [])
    if not verdicts:
        return 0.0
    supported = sum(1 for v in verdicts if v.get("supported", False))
    return supported / len(verdicts)
