import os
import pytest
from core.config import get_settings

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
