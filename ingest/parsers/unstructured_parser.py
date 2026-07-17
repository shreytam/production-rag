from __future__ import annotations

from core.types import Document
from ingest.parsers.base import ParserError


class UnstructuredParser:
    """Rich multi-format parser (PDF/Office/HTML/OCR) via `unstructured`."""

    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]:
        try:
            from unstructured.partition.auto import partition  # local heavy import
        except ImportError as e:  # pragma: no cover
            raise ParserError("unstructured not installed") from e
        import io
        elements = partition(file=io.BytesIO(raw), metadata_filename=filename)
        text = "\n\n".join(str(el) for el in elements if str(el).strip())
        if not text:
            raise ParserError(f"no extractable text in {filename}")
        return [Document(doc_id=doc_id, text=text, tenant_id=tenant_id,
                         acl_tags=acl_tags, title=filename, source=f"upload:{content_type}")]
