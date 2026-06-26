"""PII guardrail: redact PII found in user input or generated output.

Delegates detection to :func:`ingest.pii.redact` and maintains an audit log.
"""

from __future__ import annotations

from ingest.pii import PIIRedactor
from core.types import GuardrailAction, GuardrailResult


class PIIGuardrail:
    """Redacts PII using the shared :class:`ingest.pii.PIIRedactor`.

    Returns ``REDACT`` with the cleaned text in ``payload`` when PII is found,
    ``PASS`` otherwise. An internal audit log accumulates every finding for
    compliance purposes.
    """

    def __init__(self) -> None:
        self._redactor = PIIRedactor()

    @property
    def name(self) -> str:
        return "pii_guard"

    @property
    def audit_log(self) -> list[dict]:
        return self._redactor.audit_log

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        redacted, findings = self._redactor.redact(text)

        if findings:
            types = [f["type"] for f in findings]
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.REDACT,
                reason=f"PII detected: {', '.join(sorted(set(types)))}",
                payload=redacted,
                metadata={"findings": findings, "count": len(findings)},
            )

        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.PASS,
        )
