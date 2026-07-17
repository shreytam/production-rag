from __future__ import annotations

from core.types import DocumentRecord, DocumentStatus

_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    document_id  TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    status       TEXT NOT NULL,
    blob_key     TEXT NOT NULL,
    error        TEXT NOT NULL DEFAULT '',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_tenant_idx ON {table} (tenant_id);
"""


class PostgresDocumentRegistry:
    def __init__(self, dsn: str, table: str = "documents") -> None:
        self._dsn = dsn
        self._table = table
        self._ensure()

    def _conn(self):
        import psycopg  # local import
        return psycopg.connect(self._dsn)

    def _ensure(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(_DDL.format(table=self._table))

    def create(self, record: DocumentRecord) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table} (document_id, tenant_id, filename, "
                f"content_type, size_bytes, status, blob_key, error, chunk_count) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (document_id) DO NOTHING",
                [record.document_id, record.tenant_id, record.filename,
                 record.content_type, record.size_bytes, record.status.value,
                 record.blob_key, record.error, record.chunk_count])

    def get(self, document_id: str, tenant_id: str) -> DocumentRecord | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT document_id, tenant_id, filename, content_type, size_bytes, "
                f"status, blob_key, error, chunk_count FROM {self._table} "
                f"WHERE document_id=%s AND tenant_id=%s", [document_id, tenant_id])
            row = cur.fetchone()
        return self._row(row) if row else None

    def list(self, tenant_id: str) -> list[DocumentRecord]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT document_id, tenant_id, filename, content_type, size_bytes, "
                f"status, blob_key, error, chunk_count FROM {self._table} "
                f"WHERE tenant_id=%s ORDER BY created_at DESC", [tenant_id])
            rows = cur.fetchall()
        return [self._row(r) for r in rows]

    def set_status(self, document_id: str, tenant_id: str, status: DocumentStatus,
                   *, error: str = "", chunk_count: int = 0) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"UPDATE {self._table} SET status=%s, error=%s, chunk_count=%s, "
                f"updated_at=now() WHERE document_id=%s AND tenant_id=%s",
                [status.value, error, chunk_count, document_id, tenant_id])

    @staticmethod
    def _row(row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row[0], tenant_id=row[1], filename=row[2], content_type=row[3],
            size_bytes=row[4], status=DocumentStatus(row[5]), blob_key=row[6],
            error=row[7], chunk_count=row[8])
