"""Schema validation guardrail.

Validates a candidate answer dict against a Pydantic model before it is
returned to the caller. Catches hallucinated or malformed structured output
early.

Usage
-----
    result = guardrail.check(
        "",                          # text is unused; pass the dict via context
        context={"candidate": {...}},
    )

    # Or pass a raw dict directly as text (the guardrail also tries to parse
    # the text arg as JSON if context["candidate"] is absent).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from generation.grounded_generator import GeneratedAnswer
from core.types import GuardrailAction, GuardrailResult


class SchemaGuardrail:
    """Validate a candidate answer dict against *model* (default: GeneratedAnswer).

    Parameters
    ----------
    model:
        Pydantic BaseModel class to validate against. Defaults to
        :class:`generation.grounded_generator.GeneratedAnswer`.
    """

    def __init__(self, model: type[BaseModel] = GeneratedAnswer) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "schema_validation"

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        candidate: Any = context.get("candidate")

        # Fall back to parsing text as JSON if no explicit candidate provided.
        if candidate is None:
            if text:
                try:
                    candidate = json.loads(text)
                except json.JSONDecodeError as exc:
                    return GuardrailResult(
                        name=self.name,
                        action=GuardrailAction.BLOCK,
                        reason=f"Text is not valid JSON: {exc}",
                    )
            else:
                return GuardrailResult(
                    name=self.name,
                    action=GuardrailAction.BLOCK,
                    reason="No candidate dict provided and text is empty.",
                )

        try:
            self._model.model_validate(candidate)
        except ValidationError as exc:
            errors = exc.errors()
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason=f"Schema validation failed: {len(errors)} error(s).",
                metadata={"validation_errors": errors},
            )

        return GuardrailResult(name=self.name, action=GuardrailAction.PASS)
