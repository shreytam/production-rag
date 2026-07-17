from __future__ import annotations

from core.types import Document


class PlainTextParser:
    def parse(self, raw: bytes, filename: str, content_type: str, *,
              doc_id: str, tenant_id: str, acl_tags: tuple[str, ...]) -> list[Document]:
        text = raw.decode("utf-8", errors="replace")
        return [Document(doc_id=doc_id, text=text, tenant_id=tenant_id,
                         acl_tags=acl_tags, title=filename, source=f"upload:{content_type}")]
