import io

import pytest

from core.config import Settings
from ingest.parsers.base import ParserError, ParserRegistry
from ingest.parsers.pdf import PdfParser
from ingest.parsers.plain_text import PlainTextParser

DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def minimal_pdf(text: str | None) -> bytes:
    """Build a byte-correct single-page PDF (real xref table, real %%EOF).

    `text=None` produces a page with no content stream at all — the closest
    stand-in for a scanned/image-only PDF, which has no extractable text.
    """
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    ]
    if text is None:
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>")
    else:
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref_at,
    )
    return bytes(out)


def minimal_docx(text: str) -> bytes:
    import docx  # provided by python-docx, declared in the `app` extra

    buf = io.BytesIO()
    document = docx.Document()
    document.add_paragraph(text)
    document.save(buf)
    return buf.getvalue()


# One parseable sample per content type the API advertises in
# Settings.ingest_allowed_types.  test_every_advertised_upload_type_parses
# asserts this mapping stays exhaustive.
SAMPLES: dict[str, bytes] = {
    "text/plain": b"Hello PLAIN",
    "text/markdown": b"# Hello MARKDOWN",
    "application/pdf": minimal_pdf("Hello PDF"),
    DOCX_TYPE: minimal_docx("Hello DOCX"),
    "text/html": b"<html><body><p>Hello HTML</p></body></html>",
}


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


def test_pdf_parser_extracts_text():
    docs = PdfParser().parse(
        minimal_pdf("Hello PDF"), "paper.pdf", "application/pdf",
        doc_id="d1", tenant_id="t1", acl_tags=("public",),
    )
    assert len(docs) == 1
    assert "Hello PDF" in docs[0].text
    assert docs[0].doc_id == "d1"
    assert docs[0].tenant_id == "t1"
    assert docs[0].acl_tags == ("public",)
    assert docs[0].title == "paper.pdf"


def test_pdf_parser_rejects_pdf_with_no_extractable_text():
    # A scanned/image-only PDF. We do not OCR: extracting nothing must be a
    # loud ParserError, never an empty document that ingests as a silent no-op.
    with pytest.raises(ParserError) as exc_info:
        PdfParser().parse(
            minimal_pdf(None), "scan.pdf", "application/pdf",
            doc_id="d1", tenant_id="t1", acl_tags=(),
        )
    assert "scan.pdf" in str(exc_info.value)


def test_pdf_parser_rejects_bytes_that_are_not_a_pdf():
    with pytest.raises(ParserError):
        PdfParser().parse(
            b"this is not a pdf at all", "fake.pdf", "application/pdf",
            doc_id="d1", tenant_id="t1", acl_tags=(),
        )


def test_registry_resolves_pdf_to_pdf_parser():
    # PDFs must NOT route to UnstructuredParser: its PDF path imports
    # unstructured_inference (onnxruntime/opencv/torch, ~2.5 GB) at module
    # import time and downloads layout weights at first use.
    reg = ParserRegistry(allowed_types={"application/pdf"}, max_bytes=1000)
    assert isinstance(reg.resolve("application/pdf"), PdfParser)


@pytest.mark.parametrize("content_type", sorted(Settings().ingest_allowed_types.split(",")))
def test_every_advertised_upload_type_parses(content_type):
    """The allowlist is a promise: each type must resolve AND actually parse.

    This is the regression guard for the class of bug where `unstructured`
    gates a format behind optional dependencies we never declared — the API
    happily accepted the upload and the worker then failed at parse time.
    """
    assert content_type in SAMPLES, f"no sample for advertised type {content_type!r}"
    reg = ParserRegistry(allowed_types={content_type}, max_bytes=10_000_000)
    docs = reg.resolve(content_type).parse(
        SAMPLES[content_type], "sample", content_type,
        doc_id="d1", tenant_id="t1", acl_tags=(),
    )
    assert docs and docs[0].text.strip(), f"{content_type} parsed to empty text"
