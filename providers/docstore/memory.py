from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from core.types import DocumentRecord, DocumentStatus


class InMemoryDocumentRegistry:
    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._rows: dict[str, DocumentRecord] = {}
        self._updated: dict[str, datetime] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def create(self, record: DocumentRecord) -> None:
        self._rows[record.document_id] = record
        self._updated[record.document_id] = self._clock()

    def list_stale(self, statuses: tuple[DocumentStatus, ...],
                   older_than: timedelta) -> list[DocumentRecord]:
        cutoff = self._clock() - older_than
        return [r for r in self._rows.values()
                if r.status in statuses and self._updated[r.document_id] <= cutoff]

    def get(self, document_id: str, tenant_id: str) -> DocumentRecord | None:
        r = self._rows.get(document_id)
        return r if r and r.tenant_id == tenant_id else None

    def get_privileged(self, document_id: str) -> DocumentRecord | None:
        """Trusted, tenant-unscoped lookup for the worker (the row carries the
        tenant it was created under)."""
        return self._rows.get(document_id)

    def list(self, tenant_id: str) -> list[DocumentRecord]:
        return [r for r in self._rows.values() if r.tenant_id == tenant_id]

    def set_status(self, document_id: str, tenant_id: str, status: DocumentStatus,
                   *, error: str = "", chunk_count: int = 0) -> None:
        r = self.get(document_id, tenant_id)
        if r is None:
            return
        self._rows[document_id] = r.model_copy(
            update={"status": status, "error": error, "chunk_count": chunk_count})
        self._updated[document_id] = self._clock()

    def delete(self, document_id: str, tenant_id: str) -> None:
        r = self._rows.get(document_id)
        if r is not None and r.tenant_id == tenant_id:
            self._rows.pop(document_id, None)
            self._updated.pop(document_id, None)
