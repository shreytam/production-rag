from __future__ import annotations
import re
from core.types import PIISpan
from core.interfaces import PIIDetector

_PATTERNS = [
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+1[\s\-]?)?"
            r"(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])"
            r"\d{3}[\s\-]\d{4}"
            r"(?!\d)",
        ),
    ),
    (
        "SSN",
        re.compile(
            r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)",
        ),
    ),
    (
        "CREDIT_CARD",
        re.compile(
            r"(?<!\d)"
            r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"
            r"(?!\d)",
        ),
    ),
]

class RegexPIIDetector:
    def detect(self, text: str) -> list[PIISpan]:
        spans = []
        for ptype, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                spans.append(PIISpan(type=ptype, start=m.start(), end=m.end()))
        return spans
