from __future__ import annotations
import re
from core.types import PIISpan

# Unicode-aware: \w matches non-ASCII letters, so IDN domains (jo@exämple.com) hit.
_EMAIL = re.compile(
    r"(?<![\w.%+\-])[\w.%+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[^\W\d_]{2,}(?!\w)",
)

# US/NANP numbers, kept strict (separators required) to avoid matching bare digit runs.
_PHONE_US = re.compile(
    r"(?<!\d)"
    r"(?:\+1[\s\-]?)?"
    r"(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])"
    r"\d{3}[\s\-]\d{4}"
    r"(?!\d)",
)
# International E.164-style: leading '+', country code, then digit groups. The
# 8-15 total-digit bound (E.164 max is 15) is enforced in _valid_intl_phone.
_PHONE_INTL = re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[\s.\-]?\(?\d{1,5}\)?){2,6}(?!\d)")

_SSN_SEPARATED = re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)")
# Bare 9 digits are only an SSN when a context keyword immediately precedes them;
# otherwise every 9-digit identifier would be flagged. Span covers the digits only.
_SSN_CONTEXT = re.compile(
    r"(?:\bssn\b|social\s+security(?:\s+(?:number|no\.?|num\.?|#))?)(?:\s+(?:is|was))?\W{0,4}(?P<n>\d{9})(?!\d)",
    re.IGNORECASE,
)

# 13-19 digits with optional space/dash separators; brand prefix + Luhn checked below.
_CARD = re.compile(r"(?<!\d)\d(?:[\s\-]?\d){12,18}(?!\d)")
_CARD_PREFIX = re.compile(
    r"^(?:4|5[1-5]|2(?:2[2-9][1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)|3[47]|6011|65)"
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _valid_card(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    if not 13 <= len(digits) <= 19:
        return False
    if digits[:2] in ("34", "37") and len(digits) != 15:
        return False
    return bool(_CARD_PREFIX.match(digits)) and _luhn_ok(digits)


def _valid_intl_phone(raw: str) -> bool:
    return 8 <= len(re.sub(r"\D", "", raw)) <= 15


class RegexPIIDetector:
    def detect(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []

        def add(ptype: str, start: int, end: int) -> None:
            spans.append(PIISpan(type=ptype, start=start, end=end))

        for m in _EMAIL.finditer(text):
            add("EMAIL", m.start(), m.end())
        for m in _PHONE_US.finditer(text):
            add("PHONE", m.start(), m.end())
        for m in _PHONE_INTL.finditer(text):
            if not _valid_intl_phone(m.group()):
                continue
            if any(s.type == "PHONE" and m.start() < s.end and s.start < m.end() for s in spans):
                continue
            add("PHONE", m.start(), m.end())
        for m in _SSN_SEPARATED.finditer(text):
            add("SSN", m.start(), m.end())
        for m in _SSN_CONTEXT.finditer(text):
            add("SSN", m.start("n"), m.end("n"))
        for m in _CARD.finditer(text):
            if _valid_card(m.group()):
                add("CREDIT_CARD", m.start(), m.end())
        return spans
