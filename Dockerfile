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

# Install system runtime dependencies for unstructured/PDF/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and prepare /app directory owned by app
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app && \
    mkdir -p /app /app/.cache && \
    chown -R app:app /app

WORKDIR /app

USER app

# Install dependencies first so this layer is cached across source-only changes.
COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra app --frozen --no-install-project

# Now copy the source and install the project itself (no dev/test cruft —
# see .dockerignore).
COPY --chown=app:app . .
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra app --frozen

# Ensure runtime cache directories exist with correct ownership for the non-root user
RUN mkdir -p /app/.cache/uploads /app/.cache/manifest /app/.cache/sparse_tenants /app/.cache/contextual

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

