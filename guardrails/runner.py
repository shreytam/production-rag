"""GuardrailRunner: orchestrate input and output guardrails with latency tracking.

Usage
-----
    runner = default_runner(generator=my_generator)

    # Input path
    input_results = runner.check_input(user_text)
    if runner.blocked(input_results):
        raise ValueError("Input blocked")
    clean_text = runner.apply_redactions(user_text, input_results)

    # Output path
    output_results = runner.check_output(
        answer,
        context={
            "context_chunk_ids": chunk_id_set,
            "contexts": [c.chunk.text for c in answer.contexts],
            "candidate": answer.model_dump(),
        },
    )
    if runner.blocked(output_results):
        raise ValueError("Output blocked")
"""

from __future__ import annotations

import time
from typing import Any

from core.interfaces import Generator, Guardrail
from core.types import Answer, GuardrailAction, GuardrailResult


class GuardrailRunner:
    """Runs ordered lists of input and output guardrails.

    Parameters
    ----------
    input_guards:
        Guardrails applied to user input text.
    output_guards:
        Guardrails applied to generated answers.
    """

    def __init__(
        self,
        input_guards: list[Guardrail] | None = None,
        output_guards: list[Guardrail] | None = None,
    ) -> None:
        self.input_guards: list[Guardrail] = input_guards or []
        self.output_guards: list[Guardrail] = output_guards or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, text: str) -> list[GuardrailResult]:
        """Run all input guards on *text*. Returns one result per guard."""
        return [self._timed_check(guard, text) for guard in self.input_guards]

    def check_output(self, answer: Answer, context: dict[str, Any] | None = None) -> list[GuardrailResult]:
        """Run all output guards.

        Each guard receives ``answer.text`` as positional text and a merged
        context dict containing the Answer object plus caller-supplied context.
        """
        merged: dict[str, Any] = {"answer": answer}
        if context:
            merged.update(context)
        return [self._timed_check(guard, answer.text, context=merged) for guard in self.output_guards]

    @staticmethod
    def blocked(results: list[GuardrailResult]) -> bool:
        """Return True if any result has action BLOCK."""
        return any(r.action == GuardrailAction.BLOCK for r in results)

    @staticmethod
    def apply_redactions(text: str, results: list[GuardrailResult]) -> str:
        """Apply the last REDACT payload (if any) to *text*.

        Multiple REDACT results are chained: the payload of the first REDACT
        guard is fed as input to the next, and so on.
        """
        for result in results:
            if result.action == GuardrailAction.REDACT and result.payload is not None:
                text = result.payload
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_check(
        guard: Guardrail,
        text: str,
        context: dict | None = None,
    ) -> GuardrailResult:
        """Call *guard.check*, catching exceptions per the guard's fail policy.

        A guard must never 500 a request. Deterministic guards fail closed
        (BLOCK); a guard that sets ``fail_closed = False`` (groundedness) fails
        soft (PASS + ``groundedness_unverified``). Every exception is recorded.
        """
        t0 = time.perf_counter()
        try:
            result = guard.check(text, context=context)
        except Exception as e:  # noqa: BLE001 — a broken guard must not crash the request
            fail_closed = getattr(guard, "fail_closed", True)
            result = GuardrailResult(
                name=getattr(guard, "name", "unknown"),
                action=GuardrailAction.BLOCK if fail_closed else GuardrailAction.PASS,
                reason=f"guard errored: {type(e).__name__}",
                metadata={"error": str(e), "groundedness_unverified": not fail_closed},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.metadata["latency_ms"] = round(elapsed_ms, 3)
        return result


def default_runner(generator: Generator | None = None) -> GuardrailRunner:
    """Factory that wires sensible defaults for a production pipeline.

    GroundednessGuardrail is omitted when *generator* is None (offline /
    testing scenarios where no LLM is available).
    """
    from guardrails.input_injection import InjectionGuardrail
    from guardrails.pii_guard import PIIGuardrail
    from guardrails.citation_enforcement import CitationGuardrail
    from guardrails.schema_validation import SchemaGuardrail

    input_guards: list[Guardrail] = [
        InjectionGuardrail(generator=generator),  # type: ignore[arg-type]
        PIIGuardrail(),
    ]

    output_guards: list[Guardrail] = [
        CitationGuardrail(),
        SchemaGuardrail(),
    ]

    if generator is not None:
        from guardrails.output_groundedness import GroundednessGuardrail
        output_guards.append(GroundednessGuardrail(generator=generator))  # type: ignore[arg-type]

    return GuardrailRunner(input_guards=input_guards, output_guards=output_guards)
