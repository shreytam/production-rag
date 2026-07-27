-- One Postgres 17 server, two databases with separate owners.
--
--   rag       <- the app's document registry   (PG_DSN)
--   langfuse  <- the Langfuse v3 stack          (DATABASE_URL)
--
-- The two never share a role, a database, or an env var; the only thing they
-- share is the server process. The app no longer needs the pgvector extension
-- (Qdrant is the sole vector store), so the stock postgres image is enough.
--
-- IMPORTANT: files in /docker-entrypoint-initdb.d run exactly once, when the
-- data directory is EMPTY. Editing this file has no effect on a volume that has
-- already been initialised — recreate it (`docker compose down -v`, or
-- `docker volume rm <project>_pgdata17`) or apply the change by hand.
--
-- DEV-ONLY credentials. Do not reuse them in any shared deployment.

CREATE ROLE rag LOGIN PASSWORD 'rag';
CREATE DATABASE rag OWNER rag;

CREATE ROLE langfuse LOGIN PASSWORD 'langfuse';
CREATE DATABASE langfuse OWNER langfuse;

-- Postgres grants CONNECT on every database to PUBLIC by default, so without
-- this the `rag` role can open a session against the `langfuse` database and
-- vice versa. Table contents would still be protected (neither role has any
-- grant on the other's tables), but sharing one server should not silently
-- widen who can reach what. Lock each database to its own owner.
REVOKE CONNECT ON DATABASE rag      FROM PUBLIC;
GRANT  CONNECT ON DATABASE rag      TO rag;

REVOKE CONNECT ON DATABASE langfuse FROM PUBLIC;
GRANT  CONNECT ON DATABASE langfuse TO langfuse;
