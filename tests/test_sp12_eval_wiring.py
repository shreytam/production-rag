import inspect

import eval.experiment as experiment
import eval.ragas_adapter as ragas_adapter


def test_eval_enables_rewriter_g4():
    # Both eval entry points must build the pipeline with the rewriter ON so the
    # gate measures recall impact (spec G4). Guard against silent regression.
    for mod in (experiment, ragas_adapter):
        src = inspect.getsource(mod)
        assert "enable_rewriter=True" in src, f"{mod.__name__} must enable the rewriter (G4)"
