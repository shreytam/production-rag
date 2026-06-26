"""Citation enforcement guardrail.

Verifies that every citation in a generated answer is grounded in the retrieved
context chunks, and that non-refused answers include at least one citation.

Usage
-----
    result = guardrail.check(
        answer.text,
        context={
            "answer": answer,                   # core.types.Answer
            "context_chunk_ids": {"c1", "c2"},  # set[str] from retrieval
        },
    )
"""

from __future__ import annotations

from core.types import Answer, GuardrailAction, GuardrailResult


class CitationGuardrail:
    """Block answers that lack citations or contain hallucinated chunk references."""

    @property
    def name(self) -> str:
        return "citation_enforcement"

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        answer: Answer | None = context.get("answer")
        context_chunk_ids: set[str] = set(context.get("context_chunk_ids", set()))

        if answer is None:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason="No Answer object provided in context.",
            )

        # Refused answers don't need citations.
        if answer.refused:
            return GuardrailResult(name=self.name, action=GuardrailAction.PASS)

        # Non-refused answers must have at least one citation.
        if not answer.citations:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason="Non-refused answer contains zero citations.",
            )

        # Every cited chunk_id must appear in the retrieved context.
        hallucinated = [
            c.chunk_id for c in answer.citations if c.chunk_id not in context_chunk_ids
        ]
        if hallucinated:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason=f"Hallucinated citation(s): chunk_ids not in retrieved context: {hallucinated}",
                metadata={"hallucinated_chunk_ids": hallucinated},
            )

        return GuardrailResult(name=self.name, action=GuardrailAction.PASS)
