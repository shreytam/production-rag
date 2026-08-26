-- Single-instance Postgres: create the Langfuse database alongside the app's.
-- Runs only on first initialization of the pgdata volume (dev stack).
CREATE USER langfuse WITH PASSWORD 'langfuse';
CREATE DATABASE langfuse OWNER langfuse;
