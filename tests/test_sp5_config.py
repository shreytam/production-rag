import pytest
from core.config import Settings, get_settings

def test_sp5_config_defaults():
    settings = Settings()
    assert settings.judge_votes == 3
    assert settings.judge_seed == 0
    assert settings.eval_tolerance == 0.03
    assert settings.eval_fast_n == 15
    assert settings.eval_fast_seed == 0
    assert settings.eval_bootstrap_resamples == 1000
    assert settings.require_live_stores is False
