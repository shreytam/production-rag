"""Groundedness guardrail for generated answers.

Uses the same claim-extraction + verdict approach as
:func:`eval.generation_metrics.faithfulness` to score whether the answer's
claims are supported by the retrieved contexts.

Usage
-----
    result = guardrail.check(
        answer.text,
        context={
            "contexts": ["passage 1 text", "passage 2 text"],
        },
    )
"""

from __future__ import annotations

from eval.generation_metrics import faithfulness
from core.interfaces import Generator
from core.types import GuardrailAction, GuardrailResult


class GroundednessGuardrail:
    """Block answers whose faithfulness score falls below *threshold*.

    Parameters
    ----------
    generator:
        Generator used for claim extraction and verdict classification.
    threshold:
        Minimum faithfulness score (0–1) to PASS. Defaults to 0.6.
    """

    def __init__(self, generator: Generator, threshold: float = 0.6) -> None:
        self._generator = generator
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "output_groundedness"

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        contexts: list[str] = context.get("contexts", [])

        if not contexts:
            # No context to check against — pass to avoid blocking when no
            # retrieval context is available (e.g. refused answers).
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.PASS,
                reason="No contexts provided; skipping groundedness check.",
                score=None,
            )

        # Use the question as a placeholder since we operate on the answer text.
        question = context.get("question", "")
        score = faithfulness(
            question=question,
            answer=text,
            contexts=contexts,
            generator=self._generator,
        )

        if score < self.threshold:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason=f"Answer groundedness {score:.2f} below threshold {self.threshold:.2f}.",
                score=score,
            )

        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.PASS,
            score=score,
        )
