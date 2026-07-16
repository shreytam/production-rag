import sys
import subprocess

def test_run_eval_baseline_fails_without_fast():
    # Execute run_eval python command to verify validation logic
    cmd = [sys.executable, "-m", "eval.run_eval", "--dataset", "hotpotqa", "--write-baseline"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Requires --fast option when --write-baseline is active" in res.stderr
