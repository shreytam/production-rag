"""PDF text extraction via pypdf — deliberately without the OCR stack.

`unstructured`'s PDF path is not usable here: `unstructured.partition.pdf`
imports `unstructured_inference` at module scope even for the text-only "fast"
strategy, which drags in onnxruntime/opencv/torch (~2.5 GB) and downloads ONNX
layout weights from the network on first parse. pypdf is pure Python, already
in the lock file, and needs no system packages.

The trade-off is explicit: text-based PDFs extract fine; scanned/image-only
PDFs raise ParserError rather than being silently ingested as empty documents.
Adding OCR later means a new parser, not a new dependency on this one.
"""

from __future__ import annotations

import io

from core.types import Document
from ingest.parsers.base import ParserError


class PdfParser:
    """Extracts the text layer of a PDF. No OCR, no layout model, no network."""

    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError

        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
        except PyPdfError as e:
            raise ParserError(f"unreadable PDF {filename}: {e}") from e

        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text:
            raise ParserError(
                f"no extractable text in {filename}: it has no text layer "
                f"(a scanned or image-only PDF). OCR is not enabled."
            )

        return [Document(doc_id=doc_id, text=text, tenant_id=tenant_id,
                         acl_tags=acl_tags, title=filename, source=f"upload:{content_type}")]
