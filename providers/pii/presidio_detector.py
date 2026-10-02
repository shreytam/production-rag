from __future__ import annotations
from pathlib import Path
from typing import Any
from core.types import PIISpan

# Presidio's default spaCy engine self-heals a missing model by calling
# spacy.cli.download(), which shells out to `pip install <~560 MB wheel>` from
# github.com straight into the running venv. That is not acceptable on a request
# path: it needs network egress and a writable environment, it can take minutes,
# and spaCy's run_command calls sys.exit(1) on failure — so the failure arrives
# as SystemExit rather than something a caller can catch. Since this detector
# backs a *guardrail*, silently degrading that way is worse than refusing.
#
# So: pin the model explicitly and verify it is already present before building
# the engine, mirroring presidio's own condition (spacy_nlp_engine.py):
#     if not (spacy.util.is_package(name) or Path(name).exists()): download(name)
# Checking it first means that branch is never reached.
DEFAULT_SPACY_MODEL = "en_core_web_lg"

class PresidioPIIDetector:
    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL) -> None:
        self._analyzer: Any = None
        self._model_name = model_name

    def _ensure_presidio(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError:
            raise ImportError(
                "PresidioPIIDetector requires Presidio dependencies. "
                "Install them using: pip install .[pii-ner]"
            )
        import spacy

        if not (spacy.util.is_package(self._model_name) or Path(self._model_name).exists()):
            raise RuntimeError(
                f"PresidioPIIDetector needs the spaCy model {self._model_name!r}, which is "
                f"not installed. Install it ahead of time with:\n"
                f"    python -m spacy download {self._model_name}\n"
                f"(Refusing to download it now: that would pip-install a ~560 MB wheel "
                f"from the network mid-request.)"
            )

        # Pass the engine explicitly rather than letting AnalyzerEngine build a
        # default one, so the model in use is the one we just verified.
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": self._model_name}],
            }
        ).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        return self._analyzer

    def detect(self, text: str) -> list[PIISpan]:
        analyzer = self._ensure_presidio()
        results = analyzer.analyze(text=text, language="en")
        spans = []
        for r in results:
            spans.append(PIISpan(type=r.entity_type, start=r.start, end=r.end))
        return spans
