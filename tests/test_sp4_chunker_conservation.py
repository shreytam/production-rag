from core.types import Document
from ingest.chunking import chunk_document

def test_chunker_retains_all_tokens_oversized():
    # Construct a paragraph longer than max_tokens=10 with overlap=2
    para = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    doc = Document(doc_id="d1", text=para, tenant_id="t1")

    chunks = chunk_document(doc, max_tokens=10, overlap=2)
    # Check no words are truncated across slices
    combined_words = " ".join([c.text for c in chunks])
    for word in para.split():
         assert word in combined_words

def test_chunker_retains_all_tokens_sliding_overlap():
    para = "one two three four five six seven eight nine"
    doc = Document(doc_id="d1", text=para, tenant_id="t1")
    chunks = chunk_document(doc, max_tokens=5, overlap=2)
    combined_words = " ".join([c.text for c in chunks])
    for word in para.split():
         assert word in combined_words
