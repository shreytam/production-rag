from core.types import DocumentRecord, DocumentStatus
from providers.docstore.memory import InMemoryDocumentRegistry


def _rec(did="doc1", tenant="t1"):
    return DocumentRecord(document_id=did, tenant_id=tenant, filename="f.pdf",
                          content_type="application/pdf", size_bytes=10,
                          status=DocumentStatus.PROCESSING, blob_key=f"{tenant}/{did}")


def test_create_and_get():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    got = reg.get("doc1", "t1")
    assert got is not None and got.status == DocumentStatus.PROCESSING


def test_get_is_tenant_scoped():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    assert reg.get("doc1", "other") is None  # existence never leaks cross-tenant


def test_set_status_transitions():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec())
    reg.set_status("doc1", "t1", DocumentStatus.READY, chunk_count=5)
    got = reg.get("doc1", "t1")
    assert got.status == DocumentStatus.READY and got.chunk_count == 5


def test_list_scoped_to_tenant():
    reg = InMemoryDocumentRegistry()
    reg.create(_rec("d1", "t1"))
    reg.create(_rec("d2", "t2"))
    assert [r.document_id for r in reg.list("t1")] == ["d1"]
