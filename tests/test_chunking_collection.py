from core.types import Document
from ingest.chunking import chunk_document


def test_chunks_inherit_collection_id():
    doc = Document(doc_id="d", text="alpha beta gamma " * 20, tenant_id="t", collection_id="A")
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.collection_id == "A" for c in chunks)
