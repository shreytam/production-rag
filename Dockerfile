# syntax=docker/dockerfile:1
#
# Production RAG — baked application image (SP9, Docker-only scope).
#
# Serves the FastAPI app (app.api:app) by default. The SAME image also runs the
# async ingest worker via a command override, e.g.:
#
#   docker run <image> arq ingest.worker.WorkerSettings
#
# (see infra/docker-compose.yml's `api` and `ingest-worker` services).
#
# Two stages: deps are resolved with uv in `builder` and only the finished
# virtualenv is copied forward, so neither uv nor its build cache ships in the
# runtime image.

# ---------------------------------------------------------------------------
# Stage 1: builder — resolve and install dependencies into /opt/venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Pinned uv, copied from the official image. Both stages share the same Python
# base image, so the venv built here is byte-compatible with the runtime stage.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /src

# Dependency layer — invalidated only by pyproject.toml / uv.lock.
#
# --no-install-project is deliberate and load-bearing. The app is served from
# source at /app (see PYTHONPATH in the runtime stage), not installed as a
# distribution, which means:
#   * source edits never trigger a dependency resync,
#   * .dockerignore is free to exclude README.md. Installing the project would
#     run hatchling, which reads `readme = "README.md"` from pyproject.toml and
#     dies with `OSError: Readme file does not exist` on an excluded README.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra app --frozen --no-install-project --no-dev

# `unstructured` downloads an en_core_web_sm wheel from GitHub and installs it
# into site-packages on the FIRST docx/html parse (unstructured/nlp/tokenize.py
# -> _install_spacy_model, reached via partition/text_type.py). Bake it at build
# time: a user request must never trigger a network install, and site-packages
# is not writable by the non-root runtime user anyway. Version is the one
# unstructured pins in _SPACY_MODEL_VERSION.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv \
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Fail the BUILD, not a request, if the model is missing. Same lesson as the
# presidio spaCy incident: verify the model is present, never self-heal over
# the network at runtime.
RUN /opt/venv/bin/python -c "import spacy; spacy.load('en_core_web_sm')"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# No apt packages are required.
#
# An earlier draft installed poppler-utils + libgl1/libglib2.0-0/libxcb1/
# libxrender1/libxext6/libsm6. Those are the runtime libraries for pdf2image
# (poppler) and OpenCV, which are only pulled in by unstructured's hi-res OCR
# stack (unstructured-inference / onnxruntime / opencv-python). We do not
# install that stack: PDFs go through ingest/parsers/pdf.py (pypdf, pure
# Python). Shipping those libraries without it is ~100 MB of dead layer.
# Re-add them together with the OCR extras, never on their own.
#
# libmagic1 is likewise omitted: unstructured only uses it via the `python-magic`
# binding, which is not in the lock file, and ingest/parsers/unstructured_parser.py
# now passes content_type explicitly so filetype sniffing is never needed.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy the venv already owned by `app` — a post-hoc `chown -R` would duplicate
# every file in the venv into a second layer.
COPY --from=builder --chown=app:app /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . .

USER app

EXPOSE 8000

# `.cache` holds uploaded blobs, the per-tenant BM25 pickles and the ingest
# manifest. It is a named volume in compose (shared with the worker); declaring
# it keeps a plain `docker run` from writing ingest state into the container
# layer, where it would vanish on restart.
VOLUME ["/app/.cache"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
