import pytest
from core.types import Document, Chunk
from core.config import Settings
from ingest.contextual import ContextualPrefixer
import re
import shutil

class FakeContextGenerator:
    def complete(self, messages, **kwargs):
        # Simply returns a blurb quoting the chunk to mimic the contextual LLM behavior
        user_content = messages[1].content
        # Extract chunk text from mock tags
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
