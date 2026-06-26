"""Native RAGAS-style generation evaluation metrics.

Each metric accepts an injected ``generator`` (and ``embedder`` where needed)
so it can be tested fully offline with fakes.  No ``ragas`` library is used;
the metric definitions follow the original RAGAS paper.

Pydantic schemas defined here are passed as ``response_model`` to
``Generator.complete()``, which populates ``LLMResponse.parsed`` as a dict.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.interfaces import Embedder, Generator
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


class GeneratedQuestions(BaseModel):
    """Questions that the given answer would answer."""

    questions: list[str] = Field(description="Questions that the answer could answer.")


class ContextRelevance(BaseModel):
    """Relevance of a single context chunk to the question."""

    relevant: bool = Field(description="True if this context helps answer the question.")


class StatementList(BaseModel):
    """Statements extracted from a ground-truth answer."""

    statements: list[str] = Field(description="Factual statements from the ground truth.")


class StatementVerdict(BaseModel):
    """Whether a single statement is attributable to the given contexts."""

    statement: str
    attributable: bool


class StatementVerdicts(BaseModel):
    """Attribution verdicts for all statements."""

    verdicts: list[StatementVerdict]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


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


def answer_relevancy(
    question: str,
    answer: str,
    generator: Generator,
    embedder: Embedder,
    n_questions: int = 3,
) -> float:
    """Mean cosine similarity between generated reverse-questions and the original.

    The LLM generates *n_questions* questions that *answer* would answer, then
    each is embedded alongside the original *question*; the score is the mean
    cosine similarity.
    """
    gen_resp = generator.complete(
        [
            _system(
                "You are an expert at inferring what questions a given answer addresses."
            ),
            _user(
                f"Answer: {answer}\n\n"
                f"Generate exactly {n_questions} questions that this answer could answer. "
                "Return as a JSON list."
            ),
        ],
        response_model=GeneratedQuestions,
        max_tokens=512,
    )
    generated: list[str] = _parsed(gen_resp.parsed, "questions", [])
    if not generated:
        return 0.0

    orig_vec = embedder.embed_query(question)
    gen_vecs = embedder.embed_documents(generated)

    sims = [_cosine(orig_vec, gv) for gv in gen_vecs]
    return float(np.mean(sims)) if sims else 0.0


def context_precision(
    question: str,
    answer: str,
    contexts: list[str],
    generator: Generator,
) -> float:
    """RAGAS context precision: precision@k-weighted average over context ranks.

    For each context (in rank order) judge whether it is relevant for answering
    the question. Then compute:

        CP = Σ_{k=1}^{N}  (Precision@k · rel_k) / total_relevant

    where ``rel_k ∈ {0,1}`` and ``Precision@k = #relevant_in_top_k / k``.

    Returns 0.0 when no context is relevant.
    """
    if not contexts:
        return 0.0

    relevances: list[bool] = []
    for ctx in contexts:
        resp = generator.complete(
            [
                _system("You judge whether a context chunk is relevant for answering a question."),
                _user(
                    f"Question: {question}\n\nExpected answer: {answer}\n\n"
                    f"Context: {ctx}\n\n"
                    "Is this context relevant for answering the question? (true/false)"
                ),
            ],
            response_model=ContextRelevance,
            max_tokens=256,
        )
        rel: bool = _parsed(resp.parsed, "relevant", False)
        relevances.append(bool(rel))

    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0

    running_hits = 0
    precision_sum = 0.0
    for k, rel in enumerate(relevances, start=1):
        if rel:
            running_hits += 1
            precision_sum += running_hits / k

    return precision_sum / total_relevant


def context_recall(
    question: str,
    ground_truth: str,
    contexts: list[str],
    generator: Generator,
) -> float:
    """Fraction of ground-truth statements attributable to the provided contexts.

    Steps
    -----
    1. Break *ground_truth* into atomic statements.
    2. For each statement judge whether it can be attributed to *contexts*.

    Score = attributable_statements / total_statements.
    """
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1 – statement extraction
    extract_resp = generator.complete(
        [
            _system("Extract every atomic factual statement from the ground-truth answer."),
            _user(
                f"Question: {question}\n\nGround truth: {ground_truth}\n\n"
                "List all atomic factual statements as JSON."
            ),
        ],
        response_model=StatementList,
        max_tokens=512,
    )
    statements: list[str] = _parsed(extract_resp.parsed, "statements", [])
    if not statements:
        return 0.0

    # Step 2 – attribution
    stmts_block = "\n".join(f"- {s}" for s in statements)
    verdict_resp = generator.complete(
        [
            _system(
                "Determine whether each statement from the ground truth can be "
                "attributed to (is supported by) the provided contexts."
            ),
            _user(
                f"Contexts:\n{context_block}\n\nStatements:\n{stmts_block}\n\n"
                "For each statement output attributable: true/false."
            ),
        ],
        response_model=StatementVerdicts,
        max_tokens=512,
    )
    verdicts: list[dict] = _parsed(verdict_resp.parsed, "verdicts", [])
    if not verdicts:
        return 0.0
    attributable = sum(1 for v in verdicts if v.get("attributable", False))
    return attributable / len(verdicts)
