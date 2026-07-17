import pytest

from ingest.parsers.base import ParserError, ParserRegistry
from ingest.parsers.plain_text import PlainTextParser


def test_plain_text_parser_makes_one_document():
    docs = PlainTextParser().parse(
        b"hello world", "note.txt", "text/plain",
        doc_id="d1", tenant_id="t1", acl_tags=(),
    )
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].doc_id == "d1"
    assert docs[0].tenant_id == "t1"


def test_registry_rejects_disallowed_type():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    with pytest.raises(ParserError):
        reg.resolve("application/x-evil")


def test_registry_rejects_oversize():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=4)
    with pytest.raises(ParserError):
        reg.guard_size(b"12345")


def test_registry_resolves_plain_text():
    reg = ParserRegistry(allowed_types={"text/plain"}, max_bytes=1000)
    assert isinstance(reg.resolve("text/plain"), PlainTextParser)
