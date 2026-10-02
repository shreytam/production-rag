import os

# Pin test determinism BEFORE any settings import: ambient infra/.env may enable
# Langfuse, but the offline suite exercises the no-op tracer path exclusively
# (real env vars outrank dotenv files in pydantic-settings).
os.environ["LANGFUSE_ENABLED"] = "false"
# Never touch a real Redis from the offline suite.
os.environ["RATE_LIMIT_BACKEND"] = "memory"

import pytest
from core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Make every test independent of the developer's local env files / env.

    Settings reads infra/.env and .env; a dev machine's live keys would leak into
    default-value assertions (and into failure output). Disable env-file loading
    and strip every env var that maps to a Settings field.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name, field in Settings.model_fields.items():
        keys = {name, name.upper()}
        alias = field.validation_alias
        if isinstance(alias, str):
            keys.add(alias)
        elif alias is not None and hasattr(alias, "choices"):
            keys.update(c for c in alias.choices if isinstance(c, str))
        for key in keys:
            for variant in {key, key.upper(), key.lower()}:
                monkeypatch.delenv(variant, raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    """Per-test limiter state so shared tenant ids never trip a leftover window."""
    from app import ratelimit

    ratelimit._limiter = None
    yield
    ratelimit._limiter = None


@pytest.fixture
def require_live_or_fail():
    """Fail the test instead of skipping if require_live_stores config is enabled and stores are unreachable."""
    def _verify(reachable: bool, backend: str):
        if not reachable:
            settings = get_settings()
            # Check both config setting and env parameter
            if settings.require_live_stores or os.environ.get("RAG_REQUIRE_LIVE_STORES") == "1":
                pytest.fail(f"Required connection to {backend} is down/missing in this gated test run!")
            else:
                pytest.skip(f"Connection to {backend} is unreachable. Skipping live store test.")
    return _verify
