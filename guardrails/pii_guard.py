"""PII guardrail: redact PII found in user input or generated output.

Delegates detection to :func:`ingest.pii.redact` and maintains an audit log.
"""

from __future__ import annotations

from collections import deque

from ingest.pii import PIIRedactor
from core.types import GuardrailAction, GuardrailResult


AUDIT_LOG_MAXLEN = 1000


class PIIGuardrail:
    """Redacts PII using the shared :class:`ingest.pii.PIIRedactor`.

    Returns ``REDACT`` with the cleaned text in ``payload`` when PII is found,
    ``PASS`` otherwise. An internal audit log accumulates every finding for
    compliance purposes.
    """

    def __init__(self, audit_maxlen: int = AUDIT_LOG_MAXLEN) -> None:
        self._redactor = PIIRedactor()
        # Bound the in-memory log: the redactor's default list grows for the
        # whole process lifetime. Durable audit goes through PIIAuditLog.
        self._redactor.audit_log = deque(maxlen=audit_maxlen)

    @property
    def name(self) -> str:
        return "pii_guard"

    @property
    def audit_log(self) -> deque[dict]:
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
