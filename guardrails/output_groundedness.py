"""Groundedness guardrail for generated answers.

Uses the same claim-extraction + verdict approach as
:func:`eval.generation_metrics.faithfulness` to score whether the answer's
claims are supported by the retrieved contexts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

from eval.generation_metrics import faithfulness
from core.interfaces import Generator
from core.types import GuardrailAction, GuardrailResult

# Module-level, bounded pool. A with-block executor would join on __exit__ and
# defeat the timeout, so we submit here and ABANDON the future on timeout (the
# call finishes in the background — a bounded thread + double cost for that
# request — because a running future cannot be cancelled).
_GROUNDEDNESS_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="groundedness")


class GroundednessGuardrail:
    """Block answers whose faithfulness score falls below *threshold*.

    Fails SOFT (PASS + ``groundedness_unverified``) on timeout/error — a slow
    judge LLM must not mass-block real answers.
    """

    fail_closed = False

    def __init__(
        self,
        generator: Generator,
        threshold: float = 0.6,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._generator = generator
        self.threshold = threshold
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "output_groundedness"

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        contexts: list[str] = context.get("contexts", [])
        answer = context.get("answer")

        if not contexts:
            # A non-refused answer with nothing to ground against is a hallucination.
            if answer is not None and not answer.refused:
                return GuardrailResult(
                    name=self.name,
                    action=GuardrailAction.BLOCK,
                    reason="Non-refused answer has no supporting context.",
                )
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.PASS,
                reason="No contexts provided; skipping groundedness check.",
                score=None,
            )

        question = context.get("question", "")
        fut = _GROUNDEDNESS_POOL.submit(
            faithfulness,
            question=question,
            answer=text,
            contexts=contexts,
            generator=self._generator,
        )
        try:
            score = fut.result(timeout=self._timeout_seconds)
        except FTimeout:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.PASS,
                reason="groundedness check timed out",
                metadata={"groundedness_unverified": True},
            )

        if score < self.threshold:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason=f"Answer groundedness {score:.2f} below threshold {self.threshold:.2f}.",
                score=score,
            )
        return GuardrailResult(name=self.name, action=GuardrailAction.PASS, score=score)
