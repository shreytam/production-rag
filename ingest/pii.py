"""PII detection and redaction via regex patterns.

Supported patterns: email, US phone, SSN, credit card numbers.
Each match is replaced with a placeholder like ``[EMAIL]`` and recorded in a
findings list for audit purposes.

Usage
-----
    redactor = PIIRedactor()
    clean, findings = redactor.redact(text)
    # findings -> list of dicts with keys: type, value, start, end
    # redactor.audit_log accumulates all findings across calls
"""

from __future__ import annotations

import re

# Pattern definitions — order matters: more specific patterns first to avoid
# overlapping matches when iterating sequentially.

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Email: user@domain.tld
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
    # US phone: various formats, e.g. (555) 123-4567, 555-123-4567, +1 555 123 4567
    # Must NOT match standalone 9-digit SSN-like numbers.
    (
        "PHONE",
        re.compile(
            r"(?<!\d)"                     # no digit before
            r"(?:\+1[\s\-]?)?"            # optional country code
            r"(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])"  # area code: (NXX) or NXX-
            r"\d{3}[\s\-]\d{4}"           # NXX-XXXX
            r"(?!\d)",                     # no digit after
        ),
    ),
    # SSN: XXX-XX-XXXX or XXX XX XXXX (with separators only)
    (
        "SSN",
        re.compile(
            r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)",
        ),
    ),
    # Credit card: 13-19 digit number, optionally space/dash-separated in groups
    # Common Luhn-plausible patterns only (Visa/MC/Amex/Discover).
    (
        "CREDIT_CARD",
        re.compile(
            r"(?<!\d)"
            r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"  # prefix
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"                # remaining groups
            r"(?!\d)",
        ),
    ),
]


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace PII in *text* with type placeholders.

    Returns
    -------
    redacted_text:
        The input with PII replaced, e.g. ``"call [PHONE]"``.
    findings:
        List of dicts, each with keys ``type``, ``value``, ``start``, ``end``
        (positions in the *original* text). Feed into an audit log.
    """
    findings: list[dict] = []

    # Collect all matches across all patterns, then replace right-to-left so
    # offsets remain valid.
    all_matches: list[tuple[int, int, str, str]] = []  # (start, end, ptype, value)

    for ptype, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            all_matches.append((m.start(), m.end(), ptype, m.group()))

    # Sort by start position; for overlapping matches keep the longer one.
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # De-overlap: skip any match that overlaps the previous accepted one.
    accepted: list[tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, ptype, value in all_matches:
        if start < last_end:
            continue
        accepted.append((start, end, ptype, value))
        last_end = end

    # Build findings (in original-text order)
    for start, end, ptype, value in accepted:
        findings.append({"type": ptype, "value": value, "start": start, "end": end})

    # Replace right-to-left so offsets stay valid.
    result = text
    for start, end, ptype, value in reversed(accepted):
        result = result[:start] + f"[{ptype}]" + result[end:]

    return result, findings


class PIIRedactor:
    """Stateful wrapper around :func:`redact` that accumulates an audit log."""

    def __init__(self) -> None:
        self.audit_log: list[dict] = []

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """Redact PII and append findings to :attr:`audit_log`."""
        redacted, findings = redact(text)
        self.audit_log.extend(findings)
        return redacted, findings
