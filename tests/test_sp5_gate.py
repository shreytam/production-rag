import pytest
from pathlib import Path
import json
from eval import compare as compare_module
from eval.compare import compare

def test_compare_fails_closed_on_nan_or_length_mismatch(tmp_path, monkeypatch):
    # compare() resolves the "new" run from the module-level RUNS_DIR constant
    # (there is no override parameter for it, unlike baseline_file for base) —
    # point RUNS_DIR at tmp_path so the file we write is the one actually read.
    monkeypatch.setattr(compare_module, "RUNS_DIR", tmp_path)

    base_file = tmp_path / "base.json"
    new_file = tmp_path / "squad.fresh_run.results.json"

    # 1. Base results. Aggregates use the {metric: {"mean": value}} shape that
    # _extract_aggregates expects (matches the real eval run output format).
    with base_file.open("w") as f:
        json.dump({
            "aggregates": {"faithfulness": {"mean": 0.9}},
            "items": [{"generation_metrics": {"faithfulness": 0.9}}]
        }, f)

    # 2. New results (NaN metric)
    with new_file.open("w") as f:
        json.dump({
            "aggregates": {"faithfulness": {"mean": float("nan")}},
            "items": [{"generation_metrics": {"faithfulness": float("nan")}}]
        }, f)

    # Must fail because of NaN metric
    assert compare("squad", baseline_file=base_file, new_version="fresh_run", tolerance=0.03) is False
