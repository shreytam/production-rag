from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import Document


class ParserError(Exception):
    """Raised on unsupported type, oversize upload, or unparseable content."""


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]: ...


class ParserRegistry:
    def __init__(self, allowed_types: set[str], max_bytes: int) -> None:
        self._allowed = set(allowed_types)
        self._max_bytes = max_bytes

    def guard_size(self, raw: bytes) -> None:
        if len(raw) > self._max_bytes:
            raise ParserError(f"upload exceeds max_upload_bytes ({self._max_bytes})")

    def resolve(self, content_type: str) -> DocumentParser:
        if content_type not in self._allowed:
            raise ParserError(f"unsupported content_type: {content_type}")
        if content_type in ("text/plain", "text/markdown"):
            from ingest.parsers.plain_text import PlainTextParser
            return PlainTextParser()
        if content_type == "application/pdf":
            # pypdf, not unstructured: see ingest/parsers/pdf.py for why.
            from ingest.parsers.pdf import PdfParser
            return PdfParser()
        from ingest.parsers.unstructured_parser import UnstructuredParser
        return UnstructuredParser()
