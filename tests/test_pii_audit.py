import json
import pytest
from pathlib import Path
from ingest.audit import PIIAuditLog
from core.types import PIISpan
from core.config import Settings

def test_audit_logs_record_no_values(tmp_path):
    log_file = tmp_path / "pii_audit.jsonl"
    settings = Settings(
        pii_audit_log_path=str(log_file),
        pii_audit_value_hash=False
    )
    
    audit = PIIAuditLog(settings)
    spans = [
        PIISpan(type="EMAIL", start=10, end=25)
    ]
    
    audit.record(
        tenant_id="test-tenant",
        doc_id="doc-123",
        chunk_id="chunk-456",
        text="Contact at alice@corp.com please",
        spans=spans
    )
    
    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    
    record = json.loads(lines[0])
    assert record["tenant_id"] == "test-tenant"
    assert record["doc_id"] == "doc-123"
    assert record["chunk_id"] == "chunk-456"
    assert record["type"] == "EMAIL"
    assert record["start"] == 10
    assert record["end"] == 25
    assert "timestamp" in record
    # Make sure absolutely no raw value or text segment leak occurred
    assert "value" not in record
    assert "alice" not in str(record)

def test_audit_logs_record_with_hash(tmp_path):
    log_file = tmp_path / "pii_audit.jsonl"
    settings = Settings(
        pii_audit_log_path=str(log_file),
        pii_audit_value_hash=True,
        pii_audit_hash_salt="secret-salt"
    )
    
    audit = PIIAuditLog(settings)
    spans1 = [PIISpan(type="EMAIL", start=11, end=26)]  # "alice@corp.com"
    spans2 = [PIISpan(type="EMAIL", start=11, end=26)]  # "alice@corp.com"
    spans3 = [PIISpan(type="EMAIL", start=11, end=24)]  # "bob@corp.com" (mock length diff)
    
    audit.record("t1", "d1", "c1", "My email is alice@corp.com.", spans1)
    audit.record("t1", "d2", "c2", "My email is alice@corp.com.", spans2)
    audit.record("t1", "d3", "c3", "My email is bob@corp.com.", spans3)
    
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    r3 = json.loads(lines[2])
    
    # Same value must yield same hash
    assert r1["value_hash"] == r2["value_hash"]
    # Different value must yield different hash
    assert r1["value_hash"] != r3["value_hash"]
    # No raw plaintext stored
    assert "value" not in r1
    assert "alice" not in str(r1)
