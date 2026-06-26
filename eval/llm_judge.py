"""Holistic LLM-as-judge for RAG answer quality.

The judge returns a numeric score (0–1) with a textual rationale via structured
output.  The rubric covers three dimensions: groundedness, completeness, and
relevance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.interfaces import Generator
from core.types import ChatMessage


RUBRIC = """Score the response on a 0–1 scale considering:

1. Groundedness (0–1): Are all claims in the answer directly supported by the
   provided contexts?  Penalise hallucinations and unsupported statements.

2. Completeness (0–1): Does the answer address all parts of the question?
   Penalise missing important aspects that the contexts could have covered.

3. Relevance (0–1): Is the answer focused on the question and free of
   irrelevant tangents?

Final score = mean(groundedness, completeness, relevance).
Return a single overall `score` (float 0–1) and a concise `rationale` string
explaining the key strengths and weaknesses."""


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Overall quality score 0–1.")
    rationale: str = Field(description="Explanation of the score covering groundedness, completeness, and relevance.")


def holistic_judge(
    question: str,
    answer: str,
    contexts: list[str],
    generator: Generator,
) -> dict:
    """Evaluate an answer holistically using an LLM judge.

    Parameters
    ----------
    question:
        The original user question.
    answer:
        The RAG-generated answer to evaluate.
    contexts:
        The retrieved context chunks that were provided to the generator.
    generator:
        An injected ``Generator`` instance (use a fake in tests).

    Returns
    -------
    dict with keys ``score`` (float 0–1) and ``rationale`` (str).
    """
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    resp = generator.complete(
        [
            ChatMessage(role="system", content=RUBRIC),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Contexts:\n{context_block}\n\n"
                    f"Answer: {answer}\n\n"
                    "Provide your evaluation."
                ),
            ),
        ],
        response_model=JudgeOutput,
        max_tokens=512,
    )

    parsed = resp.parsed or {}
    return {
        "score": float(parsed.get("score", 0.0)),
        "rationale": str(parsed.get("rationale", "")),
    }
