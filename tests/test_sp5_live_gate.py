import pytest
from core.config import get_settings

def test_require_live_or_fail_fixture(require_live_or_fail, monkeypatch):
    # 1. Test fail path (require_live_stores = True)
    monkeypatch.setattr(get_settings(), "require_live_stores", True)
    with pytest.raises(pytest.fail.Exception):
        require_live_or_fail(False, "TestBackend")

    # 2. Test skip path (require_live_stores = False)
    monkeypatch.setattr(get_settings(), "require_live_stores", False)
    monkeypatch.delenv("RAG_REQUIRE_LIVE_STORES", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_live_or_fail(False, "TestBackend")
