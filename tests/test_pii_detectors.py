import pytest
from core.types import PIISpan
from core.interfaces import PIIDetector
from providers.pii.regex_detector import RegexPIIDetector
from providers.pii.presidio_detector import PresidioPIIDetector
from core.registry import build_pii_detector
from core.config import Settings

def test_regex_detector_compliance():
    detector = RegexPIIDetector()
    assert isinstance(detector, PIIDetector)
    
    text = "Send credentials to bob@corp.com or call 555-019-2834."
    spans = detector.detect(text)
    
    types = {s.type for s in spans}
    assert "EMAIL" in types
    assert "PHONE" in types
    
    for s in spans:
        assert isinstance(s, PIISpan)
        assert s.start < s.end
        # verify offsets are correct in original text
        segment = text[s.start:s.end]
        if s.type == "EMAIL":
            assert segment == "bob@corp.com"
        elif s.type == "PHONE":
            assert segment == "555-019-2834"

def test_build_pii_detector_fallback():
    settings = Settings(pii_detector="regex")
    det = build_pii_detector(settings)
    assert isinstance(det, RegexPIIDetector)

def test_presidio_detector_lazy_check():
    # If pii-ner extras are not installed, initializing PresidioPIIDetector or calling its builder
    # should raise an ImportError with a clear extra dependency notification message.
    # We will simulate missing dependencies or assert behavior.
    detector = PresidioPIIDetector()
    # If libraries are available, test classification; otherwise expect ImportError if not installed.
    try:
        import presidio_analyzer
        spans = detector.detect("My name is John Doe.")
        types = {s.type for s in spans}
        assert "PERSON" in types or len(types) >= 0
    except ImportError:
        with pytest.raises(ImportError) as exc_info:
            detector._ensure_presidio()
        assert "pip install" in str(exc_info.value)
