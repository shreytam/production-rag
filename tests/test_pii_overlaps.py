import pytest
from ingest.pii import redact, PIIRedactor
from core.types import PIISpan

def test_redact_overlaps_and_nesting():
    # 1. Test nesting: Bob is inside the address (12, 23) vs (12, 15)
    text = "Find Bob at 123 Main St."
    spans = [
        PIISpan(type="ADDRESS", start=12, end=23),
        PIISpan(type="PERSON", start=12, end=15)
    ]
    # Redact must sort descending and discard nested/overlapping shorter spans
    clean = redact(text, spans)
    assert clean == "Find Bob at [ADDRESS]."

    # 2. Test unsorted and overlapping inputs
    text = "aaa555-12-3456@example.org and more"
    # EMAIL overlaps with SSN inside its user sector
    spans = [
        PIISpan(type="SSN", start=3, end=14),
        PIISpan(type="EMAIL", start=0, end=26)
    ]
    clean2 = redact(text, spans)
    assert clean2 == "[EMAIL] and more"

def test_redactor_facade_value_free():
    redactor = PIIRedactor()
    text = "My SSN is 000-11-2222."
    clean, findings = redactor.redact(text)
    
    assert clean == "My SSN is [SSN]."
    assert len(findings) == 1
    # Placed values must be absent
    assert "value" not in findings[0]
    assert findings[0]["type"] == "SSN"
