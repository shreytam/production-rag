from core import config


def test_eval_gate_defaults():
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.eval_gate_mode == "bootstrap"
    assert s.eval_baseline_run == "baseline"
    assert s.eval_gate_thresholds == {}


def test_eval_gate_mode_env_override(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("EVAL_GATE_MODE", "both")
    config.get_settings.cache_clear()
    assert config.get_settings().eval_gate_mode == "both"
    config.get_settings.cache_clear()
