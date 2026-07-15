"""pgvector vector store implementation.

ACL enforcement: pg_where() generates a WHERE clause injected before the
ORDER BY embedding <=> %s, so Postgres filters rows by tenant+tags BEFORE
computing distances — no candidate outside the ACL scope is ever scored.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from core.config import Settings
from core.types import ACLContext, Chunk, RetrievalSource, ScoredChunk, Vector
from retrieval.acl import pg_where


def _as_vector(embedding: Vector) -> "np.ndarray":
    """pgvector's psycopg dumper adapts numpy arrays to the `vector` type; a plain
    Python list would be sent as `double precision[]` and fail the `<=>` operator."""
    return np.asarray(embedding, dtype=np.float32)


def _connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn)
    # The pgvector type adapter can only be registered once the extension exists,
    # so ensure + commit it before register_vector() looks the type up.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


class PgVectorStore:
    """Dense vector store backed by PostgreSQL + pgvector.

    ACL is enforced by injecting pg_where() into every SELECT before the
    ORDER BY clause, so distance computation is restricted to matching rows.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._table = settings.pg_table

    def _conn(self) -> psycopg.Connection:
        return _connect(self._settings.pg_dsn)

    def ensure_collection(self, dimension: int) -> None:
        """Create the extension, table, and index if they don't exist."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        chunk_id          TEXT PRIMARY KEY,
                        doc_id            TEXT NOT NULL,
                        tenant_id         TEXT NOT NULL,
                        acl_tags          TEXT[] NOT NULL DEFAULT '{{}}',
                        text              TEXT NOT NULL,
                        ordinal           INTEGER NOT NULL DEFAULT 0,
                        title             TEXT,
                        source            TEXT,
                        contextual_prefix TEXT,
                        metadata          JSONB NOT NULL DEFAULT '{{}}',
                        embedding         VECTOR({dimension}) NOT NULL
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self._table}_tenant_idx
                    ON {self._table} (tenant_id)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            conn.commit()

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks (upsert on chunk_id)."""
        if not chunks:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    if chunk.embedding is None:
                        raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
                    cur.execute(
                        f"""
                        INSERT INTO {self._table}
                            (chunk_id, doc_id, tenant_id, acl_tags, text,
                             ordinal, title, source, contextual_prefix, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            doc_id            = EXCLUDED.doc_id,
                            tenant_id         = EXCLUDED.tenant_id,
                            acl_tags          = EXCLUDED.acl_tags,
                            text              = EXCLUDED.text,
                            ordinal           = EXCLUDED.ordinal,
                            title             = EXCLUDED.title,
                            source            = EXCLUDED.source,
                            contextual_prefix = EXCLUDED.contextual_prefix,
                            metadata          = EXCLUDED.metadata,
                            embedding         = EXCLUDED.embedding
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.tenant_id,
                            list(chunk.acl_tags),
                            chunk.text,
                            chunk.ordinal,
                            chunk.title,
                            chunk.source,
                            chunk.contextual_prefix,
                            json.dumps(chunk.metadata),
                            _as_vector(chunk.embedding),
                        ),
                    )
            conn.commit()

    def search(
        self, embedding: Vector, top_k: int, acl: ACLContext
    ) -> list[ScoredChunk]:
        """Search with ACL filter applied before distance ordering."""
        where_sql, where_params = pg_where(acl)
        query = f"""
            SELECT
                chunk_id, doc_id, tenant_id, acl_tags, text,
                ordinal, title, source, contextual_prefix, metadata,
                embedding <=> %s AS distance
            FROM {self._table}
            WHERE {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        # Placeholder order in the SQL: SELECT-distance embedding, WHERE params,
        # ORDER BY embedding, LIMIT.
        qvec = _as_vector(embedding)
        params = [qvec] + where_params + [qvec, top_k]

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        scored: list[ScoredChunk] = []
        for rank, row in enumerate(rows, start=1):
            (
                chunk_id, doc_id, tenant_id, acl_tags, text,
                ordinal, title, source, contextual_prefix, metadata_raw,
                distance,
            ) = row
            metadata: dict[str, Any] = (
                json.loads(metadata_raw) if isinstance(metadata_raw, str)
                else (metadata_raw or {})
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                tenant_id=tenant_id,
                acl_tags=tuple(acl_tags or []),
                text=text,
                ordinal=ordinal,
                title=title,
                source=source,
                contextual_prefix=contextual_prefix,
                metadata=metadata,
            )
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=1.0 - float(distance),
                    source=RetrievalSource.DENSE,
                    rank=rank,
                )
            )
        return scored

    def count(self, acl: ACLContext | None = None) -> int:
        """Count rows, optionally scoped to an ACL context."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if acl is not None:
                    where_sql, where_params = pg_where(acl)
                    cur.execute(
                        f"SELECT COUNT(*) FROM {self._table} WHERE {where_sql}",
                        where_params,
                    )
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                result = cur.fetchone()
        return int(result[0]) if result else 0
