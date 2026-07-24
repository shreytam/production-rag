.PHONY: install up up-app up-langfuse down ingest eval gate seed demo api test lint lint-workflows fmt clean

DATASET ?= hotpotqa
RUN     ?= local
ITEMS   ?= data/eval/$(DATASET).json
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

eval:               ## Run eval experiment:  make eval DATASET=hotpotqa RUN=myrun
	uv run python -m eval.experiment --dataset $(DATASET) --version full --run-name $(RUN)

gate:               ## Gate a run vs baseline:  make gate DATASET=hotpotqa RUN=myrun
	uv run python -m eval.gate --dataset $(DATASET) --new-run $(RUN) --baseline-run baseline

seed:               ## Seed a dataset:  make seed DATASET=hotpotqa ITEMS=data/eval/hotpotqa.json
	uv run python -m eval.dataset_cli seed --dataset $(DATASET) --items $(ITEMS)

demo:               ## Launch the Streamlit demo
	uv run streamlit run app/demo.py

api:                ## Launch the FastAPI service
	uv run uvicorn app.api:app --reload

test:               ## Run the test suite
	uv run pytest

lint:               ## Lint
	uv run ruff check .

# An invalid workflow file cannot be caught by CI itself: GitHub refuses to
# create ANY job for it, so no in-CI lint step ever runs. Check it locally.
# Install with `brew install actionlint` (skipped, with a warning, if absent).
lint-workflows:     ## Lint GitHub Actions workflows (needs actionlint)
	@if command -v actionlint >/dev/null 2>&1; then \
		actionlint; \
	else \
		echo "warning: actionlint not installed - skipping workflow lint."; \
		echo "          install it with 'brew install actionlint'."; \
	fi

fmt:                ## Format
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache
