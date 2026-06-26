.PHONY: install up up-app up-langfuse down ingest eval compare demo api test lint fmt clean

DATASET ?= hotpotqa
VERSION ?= baseline
BASE    ?= baseline
NEW     ?= full
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

ingest:             ## Ingest a dataset:  make ingest DATASET=hotpotqa
	uv run python -m ingest.run --dataset $(DATASET)

eval:               ## Run eval:  make eval DATASET=hotpotqa VERSION=baseline
	uv run python -m eval.run_eval --dataset $(DATASET) --version $(VERSION)

compare:            ## Compare two versions:  make compare BASE=baseline NEW=full
	uv run python -m eval.compare --dataset $(DATASET) --base $(BASE) --new $(NEW)

demo:               ## Launch the Streamlit demo
	uv run streamlit run app/demo.py

api:                ## Launch the FastAPI service
	uv run uvicorn app.api:app --reload

test:               ## Run the test suite
	uv run pytest

lint:               ## Lint
	uv run ruff check .

fmt:                ## Format
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache eval/runs
