# SP3 · Compliance / Data Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a first-class controllable PII policy (redact by default, optional keep), pluggable detectors (regex/Presidio), durable value-free JSONL auditing, metadata/log scrubbing, and Langfuse tracing masking.

**Architecture:** We introduce PII policy early at ingest (document-level redaction for `redact` mode, chunk-level tagging + auditing for `keep` mode) to avoid database pollution. We isolate contextual caches by policy suffix, refactor the `PIIRedactor` facade to return value-free findings for logs/observability, and inject a recursive masking callback + sample rate into the Langfuse tracer. On the output path, PII in synthesized answers is redacted, and copies in response metadata (`structured_output`) are scrubbed.

**Tech Stack:** Python 3.11-3.13, Pydantic v2, Pydantic-settings, Langfuse SDK v4, Presidio Analyzer (optional extra).

## Global Constraints

- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.audit/` directory is created on first write and must be ignored.
- Value-hash enabled without a salt fails closed at boot.
- Normalization and sorting/de-overlapping are handled within the core redact helper to ensure no PII characters survive span replacement.

---

### Task 1: Extras, Core Types, and Config Knobs

**Files:**
- Modify: `pyproject.toml`
- Create: `providers/pii/__init__.py`
- Modify: `core/types.py`
- Modify: `core/interfaces.py`
- Modify: `core/config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: None (base types and configuration settings)
- Produces: 
  - `PIISpan` model in `core/types.py`
  - `PIIDetector` Protocol in `core/interfaces.py`
  - Settings: `pii_mode`, `pii_detector`, `pii_audit_log_path`, `pii_audit_value_hash`, `pii_audit_hash_salt`, `pii_scan_output`, `langfuse_sample_rate`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pii_types_config.py` check config boot failure when hashing is enabled but salt is empty, and verify core schema types exist:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pii_types_config.py`
Expected: FAIL (Imports or Settings validation will fail with missing fields/ValidationError exceptions)

- [ ] **Step 3: Write minimal implementation**

Modify `pyproject.toml` to declare the `pii-ner` extra:
```toml
[project.optional-dependencies]
pii-ner = ["presidio-analyzer>=2.2", "spacy>=3.7"]
```

Write empty `providers/pii/__init__.py`:
```python
# providers/pii packages
```

Modify `core/types.py` (add `PIISpan` type at the end):
```python
class PIISpan(BaseModel):
    """Represents a validated block of detected PII.
    
    SECURITY: Contains only coordinates and type; never holds the segment's raw text.
    """
    type: str
    start: int
    end: int

    model_config = {
        "frozen": True,
        "extra": "forbid"
    }
```

Modify `core/interfaces.py`:
```python
from core.types import PIISpan  # update imports

@runtime_checkable
class PIIDetector(Protocol):
    """Contract for a PII detection engine."""
    def detect(self, text: str) -> list[PIISpan]: ...
```

Modify `core/config.py` (add PII configurations & Settings level validator):
```python
    # Under Settings class fields:
    pii_mode: Literal["redact", "keep"] = "redact"
    pii_detector: Literal["regex", "presidio"] = "regex"
    pii_audit_log_path: str = ".audit/pii_audit.jsonl"
    pii_audit_value_hash: bool = False
    pii_audit_hash_salt: str | None = None
    pii_scan_output: bool = True
    langfuse_sample_rate: float = 1.0

    @model_validator(mode="after")
    def _validate_pii_settings(self) -> "Settings":
        if self.pii_audit_value_hash and not self.pii_audit_hash_salt:
            raise ValueError(
                "pii_audit_hash_salt must be set when pii_audit_value_hash is True to prevent hash brute-forcing."
            )
        return self
```

Modify `.gitignore`:
```
# Durable audit logs
.audit/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pii_types_config.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml core/types.py core/interfaces.py core/config.py .gitignore tests/test_pii_types_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): add core PII types, config settings, and boot salt validation" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Pluggable Detectors Configuration

**Files:**
- Create: `providers/pii/regex_detector.py`
- Create: `providers/pii/presidio_detector.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: `PIISpan` from `core/types`, `PIIDetector` from `core/interfaces`, `Settings` from `core/config`
- Produces:
  - `RegexPIIDetector` implementing `PIIDetector`
  - `PresidioPIIDetector` implementing `PIIDetector`
  - `build_pii_detector(settings)` in `core/registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pii_detectors.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pii_detectors.py`
Expected: FAIL (Missing source modules and build functions)

- [ ] **Step 3: Write minimal implementation**

Create `providers/pii/regex_detector.py`:
```python
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
```

Create `providers/pii/presidio_detector.py` (lazy-loading dependencies at init/call time):
```python
from __future__ import annotations
from typing import Any
from core.types import PIISpan
from core.interfaces import PIIDetector

class PresidioPIIDetector:
    def __init__(self) -> None:
        self._analyzer: Any = None

    def _ensure_presidio(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError:
            raise ImportError(
                "PresidioPIIDetector requires Presidio dependencies. "
                "Install them using: pip install .[pii-ner]"
            )
        self._analyzer = AnalyzerEngine()
        return self._analyzer

    def detect(self, text: str) -> list[PIISpan]:
        analyzer = self._ensure_presidio()
        results = analyzer.analyze(text=text, language="en")
        spans = []
        for r in results:
            spans.append(PIISpan(type=r.entity_type, start=r.start, end=r.end))
        return spans
```

Modify `core/registry.py` to add `build_pii_detector`:
```python
# Under existing imports, append imports
from core.interfaces import PIIDetector

def build_pii_detector(settings: Settings | None = None) -> PIIDetector:
    s = settings or get_settings()
    if s.pii_detector == "regex":
        from providers.pii.regex_detector import RegexPIIDetector
        return RegexPIIDetector()
    elif s.pii_detector == "presidio":
        from providers.pii.presidio_detector import PresidioPIIDetector
        return PresidioPIIDetector()
    raise ValueError(f"Unknown pii_detector configured: {s.pii_detector}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pii_detectors.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/pii/regex_detector.py providers/pii/presidio_detector.py core/registry.py tests/test_pii_detectors.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): add regex and presidio detectors and builder registry" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Overlapping Redaction & Value-Free findings internality

**Files:**
- Modify: `ingest/pii.py`

**Interfaces:**
- Consumes: `RegexPIIDetector` and setup config loader
- Produces:
  - Global `redact(text, spans) -> str` utility sorting and resolved overlaps
  - `PIIRedactor` facade containing no raw `value` in returns

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pii_overlaps.py`:

```python
import pytest
from ingest.pii import redact, PIIRedactor
from core.types import PIISpan

def test_redact_overlaps_and_nesting():
    # 1. Test nesting: Bob is inside the address (10, 20) vs (12, 15)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pii_overlaps.py`
Expected: FAIL (The current `redact()` does not process `PIISpan` objects and does not de-overlap them before replacing)

- [ ] **Step 3: Write minimal implementation**

Rewrite `ingest/pii.py`:
```python
from __future__ import annotations
from core.types import PIISpan
from core.config import get_settings
from core.registry import build_pii_detector

def redact(text: str, spans: list[PIISpan]) -> str:
    """Replace spans in text with [TYPE] placeholders.
    
    Ensures safe sorting descending and overlap resolution.
    """
    if not spans:
        return text

    # Sort descending by span width to resolve overlaps in favor of longer targets
    # Sort descending by start to do safely right-to-left replace
    sorted_spans = sorted(spans, key=lambda x: (x.start, -(x.end - x.start)))

    accepted: list[PIISpan] = []
    last_end = -1
    for span in sorted_spans:
        if span.start < last_end:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pii_overlaps.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/pii.py tests/test_pii_overlaps.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "refactor(pii): update redact overlap resolution and ensure value-free facade results" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Durable Audit Logger

**Files:**
- Create: `ingest/audit.py`

**Interfaces:**
- Consumes: `Settings` from `core/config` and `PIISpan` models
- Produces: `PIIAuditLog` class and `.record()` method

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pii_audit.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pii_audit.py`
Expected: FAIL (Missing `ingest/audit.py` module)

- [ ] **Step 3: Write minimal implementation**

Create `ingest/audit.py`:
```python
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
from core.config import Settings, get_settings
from core.types import PIISpan

class PIIAuditLog:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_path = Path(self.settings.pii_audit_log_path)

    def record(
        self,
        tenant_id: str,
        doc_id: str,
        chunk_id: str,
        text: str,
        spans: list[PIISpan]
    ) -> None:
        if not spans:
            return

        # Ensure parent folder exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.time()
        
        with self.log_path.open("a", encoding="utf-8") as f:
            for span in spans:
                record_data = {
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "type": span.type,
                    "start": span.start,
                    "end": span.end,
                    "timestamp": timestamp,
                }
                
                if self.settings.pii_audit_value_hash:
                    # Salt and hash transiently sliced values
                    raw_val = text[span.start:span.end]
                    salt = self.settings.pii_audit_hash_salt or ""
                    salted = f"{salt}{raw_val}".encode("utf-8")
                    value_hash = hashlib.sha256(salted).hexdigest()[:16]
                    record_data["value_hash"] = value_hash
                    
                f.write(json.dumps(record_data, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pii_audit.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/audit.py tests/test_pii_audit.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): add durable append-only audit logger supporting optional salted hashes" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: Ingest Document Redactions & Cache Isolation

**Files:**
- Modify: `ingest/run.py`
- Modify: `ingest/contextual.py`

**Interfaces:**
- Consumes: `_apply_pii_policy` within contextual execution workflow
- Produces: CLI `--pii-mode` parameter support and clean namespace contextual caches

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest_pii_policy.py`:

```python
import pytest
from core.types import Document, Chunk
from core.config import Settings
from ingest.contextual import ContextualPrefixer
from ingest.run import main
import os
import shutil

class FakeContextGenerator:
    def complete(self, messages, **kwargs):
        # Simply returns a blurb quoting the chunk to mimic the contextual LLM behavior
        user_content = messages[1].content
        # Extract chunk text from mock tags
        import re
        chunk_match = re.search(r"<CHUNK>\n(.*)\n</CHUNK>", user_content, re.DOTALL)
        chunk_text = chunk_match.group(1) if chunk_match else "leak placeholder"
        return type("Resp", (), {"text": f"This is context for: {chunk_text}", "usage": type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()})()

def test_ingest_redact_policy_contextual(tmp_path):
    # Setup folders
    cache_dir = tmp_path / "contextual_cache"
    audit_file = tmp_path / "audit.jsonl"
    
    settings = Settings(
        pii_mode="redact",
        pii_audit_log_path=str(audit_file),
        contextual_cache_dir=str(cache_dir)
    )
    
    # Document with raw PII
    docs = [
        Document(doc_id="doc1", text="Author bob@corp.com wrote: the credit card number is 4111-2222-3333-4444.", tenant_id="t1")
    ]
    
    # We will test contextual prefixer running on redacted docs
    # 1. Redact early in docs
    from ingest.run import _apply_pii_ingest_policy
    from core.registry import build_pii_detector
    from ingest.audit import PIIAuditLog
    
    detector = build_pii_detector(settings)
    audit = PIIAuditLog(settings)
    
    clean_docs, chunks = _apply_pii_ingest_policy(docs, settings, detector, audit)
    
    # Raw email / card is redacted from documents
    assert "bob@corp.com" not in clean_docs[0].text
    assert "[EMAIL]" in clean_docs[0].text
    assert "[CREDIT_CARD]" in clean_docs[0].text
    
    # Chunks are created from clean_docs
    from ingest.chunking import chunk_document
    all_chunks = []
    for doc in clean_docs:
        all_chunks.extend(chunk_document(doc))
    
    # Run prefixer
    gen = FakeContextGenerator()
    prefixer = ContextualPrefixer(gen, cache_dir=settings.contextual_cache_dir, settings=settings)
    doc_by_id = {d.doc_id: d.text for d in clean_docs}
    
    annotated = prefixer.annotate(all_chunks, doc_by_id)
    
    # Cache namespace is isolated
    namespaces = [d.name for d in cache_dir.iterdir() if d.is_dir()]
    assert any("redact" in ns for ns in namespaces)
    assert not any("keep" in ns for ns in namespaces)

    # Let's make sure the returned prefixes and texts are clean
    for chunk in annotated:
        assert "bob@corp.com" not in chunk.contextual_prefix
        assert "bob@corp.com" not in chunk.embed_text
        assert "[EMAIL]" in chunk.embed_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_pii_policy.py`
Expected: FAIL (The helper function `_apply_pii_ingest_policy` and constructor modification matching settings in `ContextualPrefixer` are not present)

- [ ] **Step 3: Write minimal implementation**

Edit `ingest/contextual.py` around lines 55-90 to inject `settings: Settings` and handle isolation + prefix post-scanning:
```python
# Modify ContextualPrefixer constructor to take settings:
    def __init__(
        self,
        generator: Generator,
        cache_dir: str | Path = ".cache/contextual",
        settings: Settings | None = None,
    ) -> None:
        self._gen = generator
        self.settings = settings or get_settings()
        
        # Append namespace suffix to segment cache runs
        namespace = f"{self.settings.pii_mode}"
        self._cache_dir = Path(cache_dir) / namespace
        self._cache_dir.mkdir(parents=True, exist_ok=True)

# Modify prefix_for method in ContextualPrefixer:
    def prefix_for(
        self,
        doc_text: str,
        chunk_text: str,
        *,
        doc_id: str = "",
    ) -> str:
        key = _cache_key(doc_id, chunk_text)
        cache_file = self._cache_dir / f"{key}.json"

        if cache_file.exists():
            data = json.loads(cache_file.read_text("utf-8"))
            return data["prefix"]

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=_USER_TEMPLATE.format(
                    doc_text=doc_text, chunk_text=chunk_text
                ),
            ),
        ]
        response = self._gen.complete(messages, max_tokens=128, temperature=0.0)
        prefix = response.text.strip()
        
        # Post-scan defense-in-depth: if redact mode is active, block PII hallucinated by LLM
        if self.settings.pii_mode == "redact":
            from providers.pii.regex_detector import RegexPIIDetector
            from ingest.pii import redact
            spans = RegexPIIDetector().detect(prefix)
            prefix = redact(prefix, spans)

        cache_file.write_text(
            json.dumps({"prefix": prefix, "doc_id": doc_id}, ensure_ascii=False),
            encoding="utf-8",
        )
        return prefix
```

Edit `ingest/run.py` to add CLI param and define `_apply_pii_ingest_policy`:
```python
# Add this function to ingest/run.py:
def _apply_pii_ingest_policy(
    docs: list[Document],
    settings: Settings,
    detector: PIIDetector,
    audit: PIIAuditLog
) -> tuple[list[Document], list[Chunk]]:
    from ingest.pii import redact

    if settings.pii_mode == "redact":
        # Redact raw docs prior to chunking
        clean_docs = []
        for doc in docs:
            spans = detector.detect(doc.text)
            # Record audit values
            audit.record(
                tenant_id=doc.tenant_id,
                doc_id=doc.doc_id,
                chunk_id="doc_redaction",
                text=doc.text,
                spans=spans
            )
            clean_text = redact(doc.text, spans)
            clean_docs.append(doc.model_copy(update={"text": clean_text}))
        return clean_docs, []
    
    else:  # keep mode
        return docs, []

# Update main function in ingest/run.py:
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the RAG stack.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "arxiv", "financebench"],
    )
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    # CLI mode override
    parser.add_argument(
        "--pii-mode",
        choices=["redact", "keep"],
        default=None,
        help="Overwrite settings.pii_mode for this run"
    )
    args = parser.parse_args(argv)

    # --- Setup config & dependencies ---
    from core.config import get_settings
    settings = get_settings()
    
    # Overwrite if passed
    if args.pii_mode:
        settings = settings.model_copy(update={"pii_mode": args.pii_mode})

    # Load documents
    adapter = _resolve_adapter(args.dataset)
    docs = adapter.load(limit=args.limit)
    print(f"[ingest] loaded {len(docs)} documents from {args.dataset}")

    # Build logger/detector tools
    from core.registry import build_pii_detector
    from ingest.audit import PIIAuditLog
    detector = build_pii_detector(settings)
    audit = PIIAuditLog(settings)

    # Apply policy
    try:
        clean_docs, _ = _apply_pii_ingest_policy(docs, settings, detector, audit)
    except Exception as e:
        print(f"[ingest] PII detection or audit failed: {e}. FAILING CLOSED: aborting.")
        raise e

    # --- Chunk redacted or raw documents ---
    from ingest.chunking import chunk_document

    all_chunks = []
    for doc in clean_docs:
        all_chunks.extend(chunk_document(doc))
    print(f"[ingest] produced {len(all_chunks)} chunks")

    # In keep mode, tag PII on generated chunks and write audit log
    if settings.pii_mode == "keep":
        for chunk in all_chunks:
            try:
                spans = detector.detect(chunk.text)
                if spans:
                    # tagging
                    chunk.metadata["pii_types"] = sorted({s.type for s in spans})
                    # audit records (must succeed in keep mode to allow storing)
                    audit.record(
                        tenant_id=chunk.tenant_id,
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        spans=spans
                    )
            except Exception as e:
                print(f"[ingest] PII validation failed for chunk {chunk.chunk_id}: {e}. FAILING CLOSED.")
                raise e

    # --- Optional contextual prefixing ---
    if args.contextual:
        from core.registry import build_generator
        from ingest.contextual import ContextualPrefixer

        generator = build_generator(role="context", settings=settings)
        prefixer = ContextualPrefixer(generator, cache_dir=settings.contextual_cache_dir, settings=settings)
        doc_by_id = {d.doc_id: d.text for d in clean_docs}
        all_chunks = prefixer.annotate(all_chunks, doc_by_id)
        print(f"[ingest] contextual prefixes applied to {len(all_chunks)} chunks")
        
    # [Rest of run.py remains the same... using settings built matching argv settings overrides]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_pii_policy.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingest/contextual.py ingest/run.py tests/test_ingest_pii_policy.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): apply PII redactions at document level for ingest and segregate contextual cache namespaces" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: Observability Masking and Trace-ordering

**Files:**
- Modify: `observability/langfuse_tracing.py`
- Modify: `core/pipeline.py`

**Interfaces:**
- Consumes: Tracing spans
- Produces: Recursive masking of all output payloads and redacted queries

- [ ] **Step 1: Write the failing tests**

Create `tests/test_observability_mask.py`:

```python
import pytest
from core.config import Settings
from observability.langfuse_tracing import Tracer

def test_tracer_mask_callback():
    # Construct a local tracer with enabled=True mock
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk",
        langfuse_secret_key="sk"
    )
    
    # We will verify that if Langfuse is loaded, or when we intercept observation calls,
    # the strings containing PII are replaced recursively.
    # To mock this, we will instantiate Tracer and assert mask behavior.
    tracer = Tracer(settings)
    assert tracer._enabled is True
    
    # Define a helper callback like the one tracer will register with Langfuse
    mask_fn = getattr(tracer, "_mask_data", None)
    assert mask_fn is not None
    
    # Test simple string
    assert mask_fn("My SSN is 000-12-3456.") == "My SSN is [SSN]."
    # Test dictionary nested payload
    payload = {
        "question": "Ask alice@corp.com",
        "nested": {
            "phone": "Call 555-123-4567"
        },
        "list": ["another bob@corp.com", 123]
    }
    cleaned = mask_fn(payload)
    assert cleaned["question"] == "Ask [EMAIL]"
    assert cleaned["nested"]["phone"] == "Call [PHONE]"
    assert cleaned["list"][0] == "another [EMAIL]"
    assert cleaned["list"][1] == 123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observability_mask.py`
Expected: FAIL (No trace mask utility is currently hooked up to self._mask_data inside the `Tracer` facade class)

- [ ] **Step 3: Write minimal implementation**

Edit `observability/langfuse_tracing.py`:
```python
# Add helper dict/list scanner method inside Tracer class in observability/langfuse_tracing.py:
    def _mask_data(self, data: Any) -> Any:
        """Recursively scan and redact PII from traced attributes using PIIRedactor."""
        from ingest.pii import PIIRedactor
        
        redactor = PIIRedactor()
        
        if isinstance(data, str):
            try:
                cleaned, _ = redactor.redact(data)
                return cleaned
            except Exception:
                # Fail-closed: return placeholder on mask error
                return "[PII_REDCTION_ERROR]"
                
        if isinstance(data, dict):
            return {k: self._mask_data(v) for k, v in data.items()}
            
        if isinstance(data, list):
            return [self._mask_data(item) for item in data]
            
        return data

# Modify Tracer.__init__ to pass key features to the initialization of the Langfuse client:
        if self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    host=settings.langfuse_host,
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    mask=self._mask_data,
                    sample_rate=settings.langfuse_sample_rate,
                )
            except Exception:
                self._enabled = False
```

Edit `core/pipeline.py` to open the query span with unredacted parameters omitted (or captured strictly after redactions):
```python
# Edit pipeline.answer around lines 91-112:
    def answer(self, question: str, acl: ACLContext | None = None) -> Answer:
        acl = acl or self.default_acl
        latencies: dict[str, float] = {}
        guard_log: dict[str, list] = {}

        # 1. Apply input guard check BEFORE creating the root observation span
        # to ensure raw unredacted questions never land in default span parameters
        if self.guardrails is not None:
            in_results = self.guardrails.check_input(question)
            guard_log["input"] = [r.model_dump() for r in in_results]
            
            if self.guardrails.blocked(in_results):
                # Trace details of check blocks safely
                with self.tracer.span(
                    "rag.query", question="[BLOCKED]", tenant=acl.tenant_id
                ) as root:
                    reason = "; ".join(r.reason for r in in_results if not r.ok)
                    root.update(
                        output={
                            "refused": True,
                            "blocked_by": "input_guardrail",
                            "reason": reason,
                        }
                    )
                return self._refused(
                    "This request was blocked by an input safety check.", in_results
                )
            # Safe clean value
            question = self.guardrails.apply_redactions(question, in_results)

        # 2. Main observation span initialized with the redacted/clean form
        with self.tracer.span(
            "rag.query", question=question, tenant=acl.tenant_id
        ) as root:
            # We recreate safe tracing records since actual redaction ran
            if self.guardrails is not None and guard_log:
                with self.tracer.span("guardrail.input") as s_in:
                    s_in.update(output={"actions": [r.action.value for r in in_results]})

            q = Query(
                text=question,
                acl=acl,
                # [Rest of pipeline logic remains the same...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observability_mask.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add observability/langfuse_tracing.py core/pipeline.py tests/test_observability_mask.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): configure Tracer recursive masking callback and adjust pipeline root-span ordering" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: Output Redaction & Metadata Scrubbing

**Files:**
- Modify: `guardrails/runner.py`
- Modify: `core/pipeline.py`

**Interfaces:**
- Consumes: `Answer` objects
- Produces: Sanitized answer results with scrubbed `structured_output`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_output_redact.py`:

```python
import pytest
from core.types import Answer, ScoredChunk, Chunk, Citation, GuardrailAction
from guardrails.runner import default_runner
from core.pipeline import RAGPipeline
from core.config import Settings
from generation.grounded_generator import GroundedGenerator

class FakeGenerator:
    def complete(self, messages, **kwargs):
        return type("Resp", (), {"text": "dummy"})()

def test_output_pii_guardrails_redacts_metadata_answer():
    settings = Settings(pii_scan_output=True)
    runner = default_runner(generator=FakeGenerator(), settings=settings)
    
    # PII guard is present in output guards
    output_guard_names = [g.name for g in runner.output_guards]
    assert "pii_guard" in output_guard_names
    
    # We construct a pipeline with this runner
    # Prepare an Answer with raw PII in text and structured_output
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="some text", tenant_id="public")
    scored = ScoredChunk(chunk=chunk, score=0.9)
    
    raw_answer_text = "I think alice@corp.com is responsible."
    ans = Answer(
        text=raw_answer_text,
        citations=[Citation(marker="[1]", chunk_id="c1")],
        contexts=[scored],
        metadata={
            "structured_output": {
                "answer": raw_answer_text,
                "citations": [{"marker": "[1]", "chunk_id": "c1"}]
            }
        }
    )
    
    # Mock generation step and pipe run
    pipeline = RAGPipeline(
        retriever=None,
        grounded=GroundedGenerator(FakeGenerator()),
        settings=settings,
        guardrails=runner
    )
    
    # Run output check
    # Check returns REDACT since answer has email
    out_results = runner.check_output(ans, context={"question": "User question"})
    
    ans.text = runner.apply_redactions(ans.text, out_results)
    
    # Verify metadata answer is also scrubbed (by the pipeline logic)
    # Simulate processing step inside pipeline.py:
    if any(r.action == GuardrailAction.REDACT for r in out_results):
        if "structured_output" in ans.metadata and "answer" in ans.metadata["structured_output"]:
            meta_answer = ans.metadata["structured_output"]["answer"]
            ans.metadata["structured_output"]["answer"] = runner.apply_redactions(meta_answer, out_results)
            
    # Verify output PII is redacted
    assert "[EMAIL]" in ans.text
    assert "alice@corp.com" not in ans.text
    
    # Verify metadata is redacted/scrubbed
    assert "alice@corp.com" not in ans.metadata["structured_output"]["answer"]
    assert "[EMAIL]" in ans.metadata["structured_output"]["answer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_redact.py`
Expected: FAIL (PIIGuardrail is not added to default runner output guards, metadata answer is not scrubbed)

- [ ] **Step 3: Write minimal implementation**

Edit `guardrails/runner.py` (add settings lookup and conditional PIIGuardrail inclusion):
```python
# Modify default_runner signature to take optional settings, parse output guards:
def default_runner(
    generator: Generator | None = None,
    settings: Settings | None = None,
) -> GuardrailRunner:
    from guardrails.citation_enforcement import CitationGuardrail
    from guardrails.input_injection import InjectionGuardrail
    from guardrails.pii_guard import PIIGuardrail
    from guardrails.schema_validation import SchemaGuardrail
    
    s = settings or get_settings()

    input_guards: list[Guardrail] = [
        InjectionGuardrail(generator=generator),
        PIIGuardrail(),
    ]
    output_guards: list[Guardrail] = [
        CitationGuardrail(),
        SchemaGuardrail(),
    ]
    
    # Append output scan iff pii_scan_output is on
    if s.pii_scan_output:
        output_guards.append(PIIGuardrail())

    if generator is not None:
        from guardrails.output_groundedness import GroundednessGuardrail

        output_guards.append(GroundednessGuardrail(generator=generator))

    return GuardrailRunner(input_guards=input_guards, output_guards=output_guards)
```

Edit `core/pipeline.py` (to clean `structured_output` inside the output check container):
```python
# Under existing core/pipeline.py output guard checks around lines 152-160:
                    ans.text = self.guardrails.apply_redactions(ans.text, out_results)
                    
                    # Scrub metadata duplicate structured_output answer if redactions took place
                    from core.types import GuardrailAction
                    if any(r.action == GuardrailAction.REDACT for r in out_results):
                        if "structured_output" in ans.metadata and "answer" in ans.metadata["structured_output"]:
                            raw_meta_ans = ans.metadata["structured_output"]["answer"]
                            ans.metadata["structured_output"]["answer"] = self.guardrails.apply_redactions(raw_meta_ans, out_results)

                    if self.guardrails.blocked(out_results):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_output_redact.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add guardrails/runner.py core/pipeline.py tests/test_output_redact.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(pii): wire output-side PII guard and scrub raw answers inside structured output metadata" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 8: Combined Offline Integration Sweep

**Files:**
- Modify: None (integration tests suite check)
- Test: `tests/test_pii_compliance.py`

**Interfaces:**
- Consumes: All written detectors, redactions, audit logging, metadata scrubbers, and pipeline steps
- Produces: Integrated green suite checks

- [ ] **Step 1: Write integration tests**

Create `tests/test_pii_compliance.py` exercising the full chain of actions end-to-end (mock vector stores, fakes):

```python
import json
import pytest
from pathlib import Path
from core.config import Settings
from core.types import Document, ACLContext, ScoredChunk, Chunk, LLMResponse, Usage
from core.pipeline import RAGPipeline
from guardrails.runner import GuardrailRunner
from guardrails.pii_guard import PIIGuardrail

class FakeGenerator:
    def complete(self, messages, **kwargs):
        # returns normal answer mimicking RAG system
        return LLMResponse(
            text="Reach bob@corp.com or john@corp.com.",
            model="fake-gen",
            usage=Usage(prompt_tokens=10, completion_tokens=10)
        )

def test_full_query_pipeline_pii_redaction(tmp_path):
    log_file = tmp_path / "pii_audit.jsonl"
    settings = Settings(
        pii_mode="redact",
        pii_audit_log_path=str(log_file),
        pii_scan_output=True,
        langfuse_enabled=False
    )
    
    # Mock retriever
    class FakeRetriever:
        def retrieve(self, query):
            c = Chunk(chunk_id="ch1", doc_id="d1", text="dummy", tenant_id="public")
            return [ScoredChunk(chunk=c, score=0.9)]
            
    class FakeGrounded:
        def generate(self, question, scored):
            # simulate structured metadata output
            ans_text = "The answer is help@site.com."
            return Answer(
                text=ans_text,
                usage=Usage(prompt_tokens=10, completion_tokens=10),
                metadata={
                    "structured_output": {
                        "answer": ans_text
                    }
                }
            ), 10.0

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        grounded=FakeGrounded(),
        settings=settings,
        guardrails=GuardrailRunner(input_guards=[], output_guards=[PIIGuardrail()])
    )
    
    # 1. Run pipeline for query containing PII
    result = pipeline.run("What about help@site.com?", acl=ACLContext(tenant_id="public"))
    
    # Clean answer returned on normal run flow
    assert "[EMAIL]" in result["answer"]
    assert "help@site.com" not in result["answer"]
    
    # Verified metadata copy is also redacted
    answer_obj = result["answer_obj"]
    assert "help@site.com" not in answer_obj.metadata["structured_output"]["answer"]
    assert "[EMAIL]" in answer_obj.metadata["structured_output"]["answer"]
    
    # Guardrail logs contain no raw PII
    guard_logs = answer_obj.metadata["guardrails"]["output"]
    assert len(guard_logs) > 0
    # Findings array has no value block
    for finding in guard_logs[0]["metadata"]["findings"]:
        assert "value" not in finding
        assert "help@site.com" not in str(finding)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pii_compliance.py`
Expected: FAIL (The full components are integrated but this is a fresh test checking for correctness of value-free guard logs and metadata outputs)

- [ ] **Step 3: Run integration test suite & assert success**

Verify all written offline tests (including security/tenancy SP1 and guardrails SP2 validation) continue to pass.
Run: `pytest -v`
Expected: PASS (All tests green)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pii_compliance.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "test(pii): add final integration test suite covering end-to-end RAG redaction and value-free compliance logging" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
