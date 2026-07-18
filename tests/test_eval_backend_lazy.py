import sys


def test_importing_eval_modules_does_not_import_langfuse():
    for m in list(sys.modules):
        if m == "langfuse" or m.startswith("langfuse."):
            del sys.modules[m]
    import importlib
    for name in ("eval.langfuse_eval", "eval.experiment", "eval.gate",
                 "eval.evaluators", "eval.dataset_cli"):
        importlib.import_module(name)
    assert "langfuse" not in sys.modules, "eval modules must not import langfuse at import time"


def test_real_backend_class_exists():
    from eval._langfuse_backend import LangfuseEvalBackend
    assert hasattr(LangfuseEvalBackend, "run_experiment")
    assert hasattr(LangfuseEvalBackend, "get_run_scores")
