import pytest
from pydantic import ValidationError
from core.types import PIISpan
from core.config import Settings

def test_pii_span_structure():
    span = PIISpan(type="EMAIL", start=1, end=5)
    assert span.type == "EMAIL"
    assert span.start == 1
    assert span.end == 5
    # Should not allow value attribute
    with pytest.raises(ValidationError):
        PIISpan(type="EMAIL", start=1, end=5, value="bob@corp.com")

def test_settings_value_hash_requires_salt():
    # If pii_audit_value_hash is True, pii_audit_hash_salt must not be empty/None
    with pytest.raises(ValidationError):
        Settings(pii_audit_value_hash=True, pii_audit_hash_salt=None)
    
    with pytest.raises(ValidationError):
        Settings(pii_audit_value_hash=True, pii_audit_hash_salt="")

    # When salt is provided, validation passes
    settings = Settings(pii_audit_value_hash=True, pii_audit_hash_salt="my-secret-salt")
    assert settings.pii_audit_value_hash is True
    assert settings.pii_audit_hash_salt == "my-secret-salt"
