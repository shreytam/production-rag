"""Task 2 (P0 ingest fixes): prove that Settings.chunk_max_tokens/chunk_overlap
actually change the chunks the CLI/eval path (ingest/run.py) produces.

ingest/run.py's main() makes real network calls (HuggingFace downloads,
embedding API, vector store upsert) and is not offline-testable end-to-end.
The chunking step, however, is a pure function of (docs, settings) --
`_chunk_documents` -- extracted from main() for exactly this reason. Testing
it directly proves the CLI path threads config the same way the worker path
does, without touching network-gated code.
"""

from __future__ import annotations

from core.config import Settings
from core.types import Document
from ingest.run import _chunk_documents


def _doc(text: str, doc_id: str = "doc-1") -> Document:
    return Document(doc_id=doc_id, text=text, tenant_id="public", acl_tags=())


# One long paragraph (no blank lines) of 2000 distinct words -- big enough
# that both the default and a much tighter token budget must split it into
# multiple chunks, so the counts are directly comparable.
_LONG_TEXT = " ".join(f"word{i}" for i in range(2000))


def test_cli_chunk_config_changes_chunk_count():
    """A non-default chunk_max_tokens/chunk_overlap in Settings must change
    the chunks _chunk_documents (the CLI/eval path) produces -- proves
    ingest/run.py no longer calls chunk_document(doc) bare with the
    hardcoded 256/32 signature defaults."""
    doc = _doc(_LONG_TEXT)

    default_chunks = _chunk_documents([doc], Settings())
    custom_chunks = _chunk_documents([doc], Settings(chunk_max_tokens=10, chunk_overlap=2))

    assert len(default_chunks) >= 1
    # A far smaller token budget must produce strictly more chunks of the
    # same document -- not just "a setting exists somewhere".
    assert len(custom_chunks) > len(default_chunks)
