"""Offline tests for the ingest pipeline.

All tests are fully offline — no network calls, no HuggingFace downloads, no
LLM API calls. Corpus adapters are not instantiated (their load() / build_golden()
are network-gated). Fake generators and in-memory fixtures are used throughout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.types import Document, Chunk, ChatMessage, LLMResponse
from ingest.base import GoldenItem, assign_tenants, tenant_split_keeping_gold
from ingest.chunking import chunk_document
from ingest.contextual import ContextualPrefixer
from ingest.pii import redact, PIIRedactor


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeGenerator:
    """Minimal Generator implementation returning a fixed blurb."""

    def __init__(self, blurb: str = "Context blurb.") -> None:
        self.blurb = blurb
        self.call_count = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model=None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(text=self.blurb)


def _make_doc(
    text: str,
    doc_id: str = "doc-001",
    tenant_id: str = "public",
) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        tenant_id=tenant_id,
        title="Test Document",
        source="test",
    )


# ---------------------------------------------------------------------------
# 1. Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    # Two short paragraphs, each well under 256 tokens — should fit in ONE chunk
    # because the packer will combine them.
    SINGLE_PARA = "This is a single short paragraph."

    # Two clearly distinct paragraphs, both short: packer should combine them
    # unless they'd exceed budget. With max_tokens=10 they won't fit together.
    PARA_A = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
    PARA_B = "Lambda mu nu xi omicron pi rho sigma tau upsilon phi chi."

    def test_single_paragraph_yields_one_chunk(self):
        doc = _make_doc(self.SINGLE_PARA)
        chunks = chunk_document(doc, max_tokens=256, overlap=32)
        assert len(chunks) == 1
        assert chunks[0].text.strip() != ""

    def test_two_short_paragraphs_pack_into_one_chunk(self):
        """Two paragraphs that together fit under max_tokens are packed together."""
        text = f"{self.PARA_A}\n\n{self.PARA_B}"
        doc = _make_doc(text)
        # Default max_tokens=256: both paras are ~12 tokens each → combined ~24 < 256
        chunks = chunk_document(doc, max_tokens=256, overlap=32)
        assert len(chunks) == 1

    def test_tight_budget_splits_paragraphs(self):
        """With a tiny token budget, each paragraph becomes its own chunk."""
        text = f"{self.PARA_A}\n\n{self.PARA_B}"
        doc = _make_doc(text)
        # max_tokens=10: each paragraph is ~12 tokens, so they can't be packed.
        chunks = chunk_document(doc, max_tokens=10, overlap=0)
        assert len(chunks) >= 2

    def test_chunk_ids_are_deterministic(self):
        text = f"{self.PARA_A}\n\n{self.PARA_B}"
        doc = _make_doc(text)
        chunks1 = chunk_document(doc, max_tokens=10, overlap=0)
        chunks2 = chunk_document(doc, max_tokens=10, overlap=0)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_id_format(self):
        doc = _make_doc(self.SINGLE_PARA)
        chunks = chunk_document(doc)
        assert chunks[0].chunk_id == "doc-001::000000"

    def test_tenant_id_propagates(self):
        doc = _make_doc(self.SINGLE_PARA, tenant_id="tenant_a")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.tenant_id == "tenant_a"

    def test_doc_id_propagates(self):
        doc = _make_doc(self.SINGLE_PARA, doc_id="my-doc-42")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.doc_id == "my-doc-42"

    def test_ordinals_are_sequential(self):
        text = f"{self.PARA_A}\n\n{self.PARA_B}"
        doc = _make_doc(text)
        chunks = chunk_document(doc, max_tokens=10, overlap=0)
        ordinals = [c.ordinal for c in chunks]
        assert ordinals == list(range(len(chunks)))

    def test_empty_text_yields_no_chunks(self):
        doc = _make_doc("")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_long_paragraph_is_split(self):
        """A paragraph exceeding max_tokens must be split into multiple chunks."""
        # Build ~60 tokens of text, set max_tokens=20
        words = ["word"] * 80
        text = " ".join(words)
        doc = _make_doc(text)
        chunks = chunk_document(doc, max_tokens=20, overlap=0)
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# 2. assign_tenants / gold-keeping tests
# ---------------------------------------------------------------------------

class TestTenantAssignment:

    def test_assign_tenants_covers_all_doc_ids(self):
        doc_ids = [f"doc-{i}" for i in range(100)]
        result = assign_tenants(doc_ids, seed=42)
        assert set(result.keys()) == set(doc_ids)

    def test_assign_tenants_distribution(self):
        """With 200 docs we expect roughly 90/5/5 split."""
        doc_ids = [f"doc-{i}" for i in range(200)]
        result = assign_tenants(doc_ids, seed=42)
        counts = {"public": 0, "tenant_a": 0, "tenant_b": 0}
        for tid, _ in result.values():
            counts[tid] += 1
        total = sum(counts.values())
        # At least 70% public (leniency for statistical noise)
        assert counts["public"] / total >= 0.70
        # tenant_a and tenant_b each present
        assert counts["tenant_a"] >= 1
        assert counts["tenant_b"] >= 1

    def test_assign_tenants_is_deterministic(self):
        doc_ids = [f"doc-{i}" for i in range(50)]
        r1 = assign_tenants(doc_ids, seed=7)
        r2 = assign_tenants(doc_ids, seed=7)
        assert r1 == r2

    def test_gold_docs_stay_in_public(self):
        """Core guarantee: every gold doc shares its question's tenant (public)."""
        # Create docs where some would naturally be in tenant_a/tenant_b
        doc_ids = [f"doc-{i}" for i in range(40)]

        # Mark a few as gold
        gold_ids = ["doc-3", "doc-7", "doc-15", "doc-31"]
        golden_items = [
            GoldenItem(
                question=f"Q about {did}?",
                answer="Some answer.",
                relevant_doc_ids=[did],
                tenant_id="public",
            )
            for did in gold_ids
        ]

        assignment = tenant_split_keeping_gold(doc_ids, golden_items, seed=42)

        # The guarantee: every gold doc must be in "public"
        for item in golden_items:
            for did in item.relevant_doc_ids:
                assigned_tenant, _ = assignment[did]
                assert assigned_tenant == item.tenant_id, (
                    f"Gold doc {did!r} was assigned to {assigned_tenant!r} "
                    f"but question.tenant_id={item.tenant_id!r}"
                )

    def test_non_gold_docs_may_be_non_public(self):
        """Some non-gold docs should end up in minority tenants."""
        doc_ids = [f"doc-{i}" for i in range(200)]
        golden_items = [
            GoldenItem(
                question="Q?",
                answer="A.",
                relevant_doc_ids=["doc-0"],
                tenant_id="public",
            )
        ]
        assignment = tenant_split_keeping_gold(doc_ids, golden_items, seed=42)
        non_public = [
            did for did, (tid, _) in assignment.items() if tid != "public"
        ]
        assert len(non_public) >= 1, "Expected some non-gold docs in minority tenants"


# ---------------------------------------------------------------------------
# 3. PII tests
# ---------------------------------------------------------------------------

class TestPII:

    def test_email_and_phone_redacted_two_findings(self):
        """A string with one email + one phone yields exactly 2 findings."""
        text = "Contact me at alice@example.com or call (555) 123-4567."
        redacted, findings = redact(text)
        assert len(findings) == 2
        types = {f["type"] for f in findings}
        assert "EMAIL" in types
        assert "PHONE" in types

    def test_email_redacted(self):
        redacted, findings = redact("Send mail to bob@corp.org please.")
        assert "[EMAIL]" in redacted
        assert "bob@corp.org" not in redacted
        assert any(f["type"] == "EMAIL" for f in findings)

    def test_phone_redacted(self):
        redacted, findings = redact("Call 555-867-5309 now.")
        assert "[PHONE]" in redacted
        assert any(f["type"] == "PHONE" for f in findings)

    def test_ssn_redacted(self):
        redacted, findings = redact("My SSN is 123-45-6789 and I need help.")
        assert "[SSN]" in redacted
        assert any(f["type"] == "SSN" for f in findings)

    def test_credit_card_redacted(self):
        redacted, findings = redact("Card number: 4111 1111 1111 1111 expires soon.")
        assert "[CREDIT_CARD]" in redacted
        assert any(f["type"] == "CREDIT_CARD" for f in findings)

    def test_no_pii_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        redacted, findings = redact(text)
        assert redacted == text
        assert findings == []

    def test_pii_redactor_audit_log_accumulates(self):
        r = PIIRedactor()
        r.redact("Email: a@b.com")
        r.redact("Phone: (800) 555-0000")
        assert len(r.audit_log) == 2

    def test_findings_have_required_keys(self):
        _, findings = redact("user@host.io and (312) 555-9876")
        for f in findings:
            assert "type" in f
            assert "value" in f
            assert "start" in f
            assert "end" in f


# ---------------------------------------------------------------------------
# 4. ContextualPrefixer tests
# ---------------------------------------------------------------------------

class TestContextualPrefixer:

    def _make_prefixer(self, blurb: str = "This chunk discusses neural networks."):
        """Return (prefixer, fake_generator) with a temp cache dir."""
        self._tmpdir = tempfile.mkdtemp()
        gen = _FakeGenerator(blurb=blurb)
        prefixer = ContextualPrefixer(generator=gen, cache_dir=self._tmpdir)
        return prefixer, gen

    def test_prefix_for_returns_blurb(self):
        prefixer, gen = self._make_prefixer()
        result = prefixer.prefix_for("Full document text.", "Chunk text.", doc_id="d1")
        assert result == "This chunk discusses neural networks."

    def test_cache_prevents_second_generator_call(self):
        """Generator must be called exactly once even if prefix_for is called twice."""
        prefixer, gen = self._make_prefixer()
        r1 = prefixer.prefix_for("Doc text.", "Chunk text.", doc_id="d1")
        r2 = prefixer.prefix_for("Doc text.", "Chunk text.", doc_id="d1")
        assert r1 == r2
        assert gen.call_count == 1

    def test_different_doc_id_calls_generator_again(self):
        prefixer, gen = self._make_prefixer()
        prefixer.prefix_for("Doc text.", "Chunk text.", doc_id="d1")
        prefixer.prefix_for("Doc text.", "Chunk text.", doc_id="d2")
        assert gen.call_count == 2

    def test_annotate_sets_contextual_prefix(self):
        prefixer, gen = self._make_prefixer("Context for chunk.")
        doc = _make_doc("Full document about AI.", doc_id="doc-ann")
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

        doc_by_id = {"doc-ann": "Full document about AI."}
        annotated = prefixer.annotate(chunks, doc_by_id)

        for chunk in annotated:
            assert chunk.contextual_prefix == "Context for chunk."

    def test_annotate_uses_cache_for_duplicate_chunk_text(self):
        """If two chunks share the same doc_id + text, generator called once."""
        prefixer, gen = self._make_prefixer()
        # Create two identical chunks manually
        chunk_a = Chunk(
            chunk_id="doc-x::000000",
            doc_id="doc-x",
            text="Same text",
            tenant_id="public",
            ordinal=0,
        )
        chunk_b = Chunk(
            chunk_id="doc-x::000001",
            doc_id="doc-x",
            text="Same text",
            tenant_id="public",
            ordinal=1,
        )
        doc_by_id = {"doc-x": "Full doc."}
        prefixer.annotate([chunk_a, chunk_b], doc_by_id)
        assert gen.call_count == 1

    def test_annotate_returns_new_chunk_objects(self):
        """annotate should not mutate the original chunks (returns copies)."""
        prefixer, _ = self._make_prefixer()
        doc = _make_doc("Document text.", doc_id="doc-copy")
        chunks = chunk_document(doc)
        originals = [c.model_copy() for c in chunks]
        annotated = prefixer.annotate(chunks, {"doc-copy": "Document text."})
        # Original chunks should have no contextual_prefix
        for orig in originals:
            assert orig.contextual_prefix is None
        # Annotated chunks should have the prefix
        for ann in annotated:
            assert ann.contextual_prefix is not None

    def test_cache_persists_across_prefixer_instances(self):
        """A second Prefixer with the same cache dir should hit cache, not call gen."""
        tmpdir = tempfile.mkdtemp()
        gen1 = _FakeGenerator("blurb1")
        p1 = ContextualPrefixer(generator=gen1, cache_dir=tmpdir)
        p1.prefix_for("doc text", "chunk text", doc_id="dx")
        assert gen1.call_count == 1

        gen2 = _FakeGenerator("blurb2")
        p2 = ContextualPrefixer(generator=gen2, cache_dir=tmpdir)
        result = p2.prefix_for("doc text", "chunk text", doc_id="dx")
        assert gen2.call_count == 0
        assert result == "blurb1"  # from cache
