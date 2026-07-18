from core.types import Chunk, Document, DocumentRecord, DocumentStatus, Query
from core.types import ACLContext


def test_collection_id_defaults_empty():
    assert Document(doc_id="d", text="x", tenant_id="t").collection_id == ""
    assert Chunk(chunk_id="c", doc_id="d", text="x", tenant_id="t").collection_id == ""
    assert DocumentRecord(document_id="d", tenant_id="t", filename="f", content_type="text/plain",
                          size_bytes=1, status=DocumentStatus.PROCESSING, blob_key="k").collection_id == ""


def test_query_collection_id_optional():
    q = Query(text="hi", acl=ACLContext(tenant_id="t"))
    assert q.collection_id is None
    assert Query(text="hi", acl=ACLContext(tenant_id="t"), collection_id="col").collection_id == "col"


def test_deleting_status_exists():
    assert DocumentStatus.DELETING.value == "deleting"
