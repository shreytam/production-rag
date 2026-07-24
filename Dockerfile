# Production RAG — baked application image (SP9, Docker-only scope).
#
# Serves the FastAPI app (app.api:app) by default. The SAME image also runs
# the async ingest worker via a command override, e.g.:
#
#   docker run <image> uv run --no-sync arq ingest.worker.WorkerSettings
#
# (see infra/docker-compose.yml's `api` and `ingest-worker` services).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so this layer is cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra app --frozen --no-install-project

# Now copy the source and install the project itself (no dev/test cruft —
# see .dockerignore).
COPY . .
RUN uv sync --extra app --frozen

# Run as non-root.
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app && \
    chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
