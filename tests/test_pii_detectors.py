import sys

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

def test_presidio_detector_reports_missing_extra(monkeypatch):
    # Without the pii-ner extra, _ensure_presidio must raise ImportError naming the
    # extra to install. Setting the module to None in sys.modules makes the import
    # fail the same way a missing install would, without uninstalling anything.
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    detector = PresidioPIIDetector()
    with pytest.raises(ImportError) as exc_info:
        detector._ensure_presidio()
    assert "pii-ner" in str(exc_info.value)


def test_presidio_detector_never_downloads_a_model(monkeypatch):
    """Presidio's default spaCy engine calls spacy.cli.download() — a live
    `pip install` of a ~560 MB wheel from GitHub — when the model is absent. That
    is unacceptable in a request path: it needs network egress and a writable
    venv, and spaCy's run_command calls sys.exit(1) on failure, surfacing as an
    uncatchable SystemExit. We must pre-empt it with an actionable error instead.
    """
    pytest.importorskip("presidio_analyzer")
    import spacy
    import spacy.cli

    # Make the model look absent, mirroring presidio's own condition:
    #   if not (spacy.util.is_package(name) or Path(name).exists()): download(name)
    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)

    def _forbidden_download(*args, **kwargs):
        raise AssertionError(
            "spacy.cli.download() was called at runtime — the detector must refuse "
            "to fetch a model instead of pip-installing one mid-request."
        )

    monkeypatch.setattr(spacy.cli, "download", _forbidden_download)

    detector = PresidioPIIDetector()
    with pytest.raises(RuntimeError) as exc_info:
        detector.detect("My name is John Doe.")

    message = str(exc_info.value)
    assert "en_core_web_lg" in message          # names the model it needs
    assert "spacy download" in message          # tells the operator how to fix it


def test_presidio_detector_detects_person_when_model_installed():
    # Runs only where the spaCy model is already present, so the "offline" suite
    # never depends on ambient machine state (this passed locally but failed in CI
    # for exactly that reason before the download guard existed).
    pytest.importorskip("presidio_analyzer")
    import spacy

    model = "en_core_web_lg"
    if not spacy.util.is_package(model):
        pytest.skip(f"spaCy model {model} not installed: python -m spacy download {model}")

    spans = PresidioPIIDetector().detect("My name is John Doe.")
    assert "PERSON" in {s.type for s in spans}
    for s in spans:
        assert isinstance(s, PIISpan)
        assert s.start < s.end


# --- Regex detector gap regressions ---------------------------------------------

def _found(text):
    d = RegexPIIDetector()
    return [(s.type, text[s.start:s.end]) for s in d.detect(text)]


@pytest.mark.parametrize("text,ptype,seg", [
    ("My SSN is 123456789.", "SSN", "123456789"),
    ("social security number: 123456789", "SSN", "123456789"),
    ("Call +44 20 7946 0958 today", "PHONE", "+44 20 7946 0958"),
    ("card 378282246310005 on file", "CREDIT_CARD", "378282246310005"),
    ("card 3782 822463 10005", "CREDIT_CARD", "3782 822463 10005"),
    ("card 4111 1111 1111 1111", "CREDIT_CARD", "4111 1111 1111 1111"),
    ("mail jo@exämple.com now", "EMAIL", "jo@exämple.com"),
])
def test_regex_detector_gap_positives(text, ptype, seg):
    assert (ptype, seg) in _found(text)


@pytest.mark.parametrize("text,ptype", [
    ("Order id 123456789 shipped", "SSN"),            # bare 9 digits, no context
    ("card 4111 1111 1111 1112", "CREDIT_CARD"),      # fails Luhn
    ("tracking 4012345678901234", "CREDIT_CARD"),     # fails Luhn
    ("card 378282246310006", "CREDIT_CARD"),          # Amex, fails Luhn
    ("ref +1234", "PHONE"),                           # too few digits
    ("version 2.0.1 at 12:30", "PHONE"),
])
def test_regex_detector_gap_negatives(text, ptype):
    assert ptype not in {t for t, _ in _found(text)}
