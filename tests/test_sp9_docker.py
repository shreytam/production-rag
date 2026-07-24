"""Infra guards for SP9 (deployability — Docker only).

Lightweight checks that the baked Docker image and its compose wiring exist
and look sane. These do not require a Docker daemon; `docker compose config`
/ `docker build` are exercised manually (see docs), not in this suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE_FILE = REPO_ROOT / "infra" / "docker-compose.yml"


def test_dockerfile_exists():
    assert DOCKERFILE.is_file()


def test_dockerignore_exists():
    assert DOCKERIGNORE.is_file()


def test_dockerfile_runs_api_with_uvicorn():
    text = DOCKERFILE.read_text()
    assert "app.api:app" in text
    assert "uvicorn" in text


def test_dockerfile_runs_as_non_root_user():
    text = DOCKERFILE.read_text()
    # A non-root USER instruction must exist, and it must not switch back to root.
    user_lines = [
        line.split(None, 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith("USER ")
    ]
    assert user_lines, "Dockerfile must set a USER"
    assert user_lines[-1] not in ("root", "0"), "final USER must not be root"


def _load_compose():
    """Return the parsed compose file, preferring PyYAML.

    Falls back to a tolerant text-based reader (raw text) if PyYAML isn't
    importable in this environment, per SP9 spec.
    """
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(COMPOSE_FILE.read_text())


def test_compose_has_api_service_with_build_and_port():
    data = _load_compose()
    if data is not None:
        services = data.get("services", {})
        assert "api" in services, "expected an `api` service in docker-compose.yml"
        api = services["api"]

        build = api.get("build")
        assert build is not None, "`api` service must build an image"
        if isinstance(build, dict):
            dockerfile = build.get("dockerfile", "Dockerfile")
        else:
            dockerfile = "Dockerfile"
        assert "Dockerfile" in str(dockerfile)

        ports = [str(p) for p in api.get("ports", [])]
        assert any(p.split(":")[-2:] == ["8000", "8000"] or "8000:8000" in p for p in ports), (
            f"expected api service to publish 8000:8000, got {ports}"
        )

        # Shared cache volume must be mounted on both `api` and `ingest-worker`.
        ingest_worker = services.get("ingest-worker", {})
        api_volumes = api.get("volumes", [])
        worker_volumes = ingest_worker.get("volumes", [])

        def _volume_names(volumes):
            names = set()
            for v in volumes:
                if isinstance(v, str):
                    names.add(v.split(":")[0])
                elif isinstance(v, dict):
                    names.add(v.get("source"))
            return names

        shared = _volume_names(api_volumes) & _volume_names(worker_volumes)
        top_level_volumes = set((data.get("volumes") or {}).keys())
        named_shared = shared & top_level_volumes
        assert named_shared, (
            "expected a named volume shared between `api` and `ingest-worker` "
            f"(api={_volume_names(api_volumes)}, worker={_volume_names(worker_volumes)}, "
            f"top-level volumes={top_level_volumes})"
        )
    else:
        # Tolerant text-based fallback if PyYAML isn't importable.
        text = COMPOSE_FILE.read_text()
        assert re.search(r"^\s*api:\s*$", text, re.MULTILINE), "expected an `api:` service block"
        assert "dockerfile: Dockerfile" in text
        assert '"8000:8000"' in text or "8000:8000" in text
        assert "app_shared_cache" in text
        # The shared volume name must appear under both service blocks, not just once.
        assert text.count("app_shared_cache") >= 3  # top-level decl + api + ingest-worker


def test_compose_shared_cache_volume_declared_top_level():
    data = _load_compose()
    if data is None:
        pytest.skip("PyYAML not importable in this environment; text fallback covers this")
    top_level_volumes = data.get("volumes") or {}
    assert "app_shared_cache" in top_level_volumes


def test_app_api_module_is_importable_as_asgi_entrypoint():
    """Guards the exact import path the Dockerfile CMD relies on."""
    import importlib

    try:
        module = importlib.import_module("app.api")
    except ImportError as exc:
        pytest.xfail(f"app.api requires an optional dep not installed here: {exc}")
        return
    assert hasattr(module, "app"), "app.api must expose an ASGI `app` object"
