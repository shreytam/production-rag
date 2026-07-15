from __future__ import annotations
from core.types import PIISpan
from core.registry import build_pii_detector

def redact(text: str, spans: list[PIISpan]) -> str:
    """Replace spans in text with [TYPE] placeholders.
    
    Ensures safe sorting descending and overlap resolution.
    """
    if not spans:
        return text

    # Sort ascending by start; for ties, longer spans first.
    sorted_spans = sorted(spans, key=lambda x: (x.start, -(x.end - x.start)))

    accepted: list[PIISpan] = []
    last_end = -1
    for span in sorted_spans:
        if span.start < last_end:
            # Overlaps the last accepted span. If it extends further, merge by
            # widening that span instead of dropping it — dropping would leave
            # the non-overlapping tail of this span unredacted.
            if span.end > last_end:
                prev = accepted[-1]
                accepted[-1] = PIISpan(type=prev.type, start=prev.start, end=span.end)
                last_end = span.end
            continue
        accepted.append(span)
        last_end = span.end

    # Replace right-to-left
    result = text
    for span in reversed(accepted):
        result = result[:span.start] + f"[{span.type}]" + result[span.end:]
    return result

class PIIRedactor:
    """Facade matching signature of previous target, return value-free findings."""
    def __init__(self) -> None:
        self.detector = build_pii_detector()
        self.audit_log: list[dict] = []

    def redact(self, text: str) -> tuple[str, list[dict]]:
        spans = self.detector.detect(text)
        cleaned = redact(text, spans)
        
        # Build value-free findings
        findings = []
        for span in spans:
            findings.append({
                "type": span.type,
                "start": span.start,
                "end": span.end
            })
        self.audit_log.extend(findings)
        return cleaned, findings
