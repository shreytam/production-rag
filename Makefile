.PHONY: install up up-app up-langfuse down ingest api console test lint fmt clean

COMPOSE := docker compose -f infra/docker-compose.yml

install:            ## Create venv and install all extras
	uv sync --all-extras

up:                 ## Start everything (app + full Langfuse v3 stack, 8 containers)
	$(COMPOSE) up -d

up-app:             ## Start app backends only (Qdrant + Postgres/pgvector)
	$(COMPOSE) up -d qdrant postgres

up-langfuse:        ## Start the Langfuse v3 stack (web+worker+db+clickhouse+redis+minio)
	$(COMPOSE) up -d langfuse-web langfuse-worker langfuse-db clickhouse redis minio

down:               ## Stop backends
	$(COMPOSE) down

ingest:             ## Bulk-ingest PDF documents:  make ingest INPUT=path/to/pdfs
	uv run python -m ingest.run --input $(INPUT)

api:                ## Launch the FastAPI service
	uv run uvicorn app.api:app --reload

console:            ## Launch the API with the test console at http://127.0.0.1:8000/ui
	AUTH_DEV_SIGNER_ENABLED=true \
	JWT_SECRET=$${JWT_SECRET:-dev-console-secret-not-for-production!} \
	uv run uvicorn app.api:app --reload

test:               ## Run the test suite
	uv run pytest

lint:               ## Lint
	uv run ruff check .

fmt:                ## Format
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache
