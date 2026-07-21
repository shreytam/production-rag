import pytest

from eval.gate import evaluate_gate
from eval.langfuse_eval import RunItemScores
from tests.eval.fake_backend import FakeEvalBackend


def _seed_runs(base_vals, new_vals, metric="recall_at_5"):
    be = FakeEvalBackend()
    be.runs[("ds", "baseline")] = [
        RunItemScores(item_id=str(i), scores={metric: v}) for i, v in enumerate(base_vals)
    ]
    be.runs[("ds", "new")] = [
        RunItemScores(item_id=str(i), scores={metric: v}) for i, v in enumerate(new_vals)
    ]
    return be


def test_bootstrap_passes_when_equal():
    be = _seed_runs([0.8] * 10, [0.8] * 10)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="bootstrap",
                         tolerance=0.03, resamples=200) is True


def test_bootstrap_fails_on_regression():
    be = _seed_runs([0.9] * 10, [0.5] * 10)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="bootstrap",
                         tolerance=0.03, resamples=200) is False


def test_threshold_fails_below_floor():
    be = _seed_runs([0.9] * 5, [0.6] * 5)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="threshold",
                         thresholds={"recall_at_5": 0.7}, resamples=200) is False


def test_threshold_passes_above_floor():
    be = _seed_runs([0.9] * 5, [0.8] * 5)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="threshold",
                         thresholds={"recall_at_5": 0.7}, resamples=200) is True


def test_both_fails_when_only_threshold_fails():
    be = _seed_runs([0.9] * 8, [0.88] * 8)
    # Diff -0.02; CI hi -0.02 is NOT < -0.03, so bootstrap passes
    # New mean 0.88 < 0.9 floor, so threshold fails
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="both",
                         tolerance=0.03, thresholds={"recall_at_5": 0.9},
                         resamples=200) is False


def test_both_fails_when_only_bootstrap_fails():
    be = _seed_runs([0.9] * 8, [0.8] * 8)
    # Diff -0.1; CI hi -0.1 < -0.03, so bootstrap fails
    # New mean 0.8 >= 0.7 floor, so threshold passes
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="both",
                         tolerance=0.03, thresholds={"recall_at_5": 0.7},
                         resamples=200) is False


def test_item_set_mismatch_raises():
    be = FakeEvalBackend()
    be.runs[("ds", "baseline")] = [RunItemScores(item_id="0", scores={"m": 0.5})]
    be.runs[("ds", "new")] = [RunItemScores(item_id="1", scores={"m": 0.5})]
    with pytest.raises(ValueError):
        evaluate_gate(backend=be, dataset="ds", new_run="new",
                      baseline_run="baseline", mode="bootstrap", resamples=50)


def test_threshold_mode_without_thresholds_raises():
    be = _seed_runs([0.9] * 5, [0.8] * 5)
    with pytest.raises(ValueError):
        evaluate_gate(backend=be, dataset="ds", new_run="new",
                      baseline_run="baseline", mode="threshold",
                      thresholds={}, resamples=50)


def test_both_mode_without_thresholds_raises():
    be = _seed_runs([0.9] * 5, [0.8] * 5)
    with pytest.raises(ValueError):
        evaluate_gate(backend=be, dataset="ds", new_run="new",
                      baseline_run="baseline", mode="both",
                      thresholds={}, resamples=50)


def test_bootstrap_mode_ignores_empty_thresholds():
    be = _seed_runs([0.8] * 10, [0.8] * 10)
    assert evaluate_gate(backend=be, dataset="ds", new_run="new",
                         baseline_run="baseline", mode="bootstrap",
                         thresholds={}, resamples=200) is True


def test_metric_only_in_new_run_raises():
    be = FakeEvalBackend()
    be.runs[("ds", "baseline")] = [RunItemScores(item_id="0", scores={"m": 0.5})]
    be.runs[("ds", "new")] = [RunItemScores(item_id="0", scores={"other": 0.5})]
    with pytest.raises(ValueError):
        evaluate_gate(backend=be, dataset="ds", new_run="new",
                      baseline_run="baseline", mode="bootstrap", resamples=50)
