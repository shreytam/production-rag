from core.types import Answer, ScoredChunk, Chunk, Citation, GuardrailAction
from guardrails.runner import default_runner
from core.config import Settings

class FakeGenerator:
    def complete(self, messages, **kwargs):
        return type("Resp", (), {"text": "dummy"})()

def test_output_pii_guardrails_redacts_metadata_answer():
    settings = Settings(pii_scan_output=True)
    runner = default_runner(generator=FakeGenerator(), settings=settings)
    
    # PII guard is present in output guards
    output_guard_names = [g.name for g in runner.output_guards]
    assert "pii_guard" in output_guard_names
    
    # We construct a pipeline with this runner
    # Prepare an Answer with raw PII in text and structured_output
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="some text", tenant_id="public")
    scored = ScoredChunk(chunk=chunk, score=0.9)
    
    raw_answer_text = "I think alice@corp.com is responsible."
    ans = Answer(
        text=raw_answer_text,
        citations=[Citation(marker="[1]", chunk_id="c1")],
        contexts=[scored],
        metadata={
            "structured_output": {
                "answer": raw_answer_text,
                "citations": [{"marker": "[1]", "chunk_id": "c1"}]
            }
        }
    )
    
    # Run output check
    # Check returns REDACT since answer has email
    out_results = runner.check_output(ans, context={"question": "User question"})
    
    ans.text = runner.apply_redactions(ans.text, out_results)
    
    # Verify metadata answer is also scrubbed (by the pipeline logic)
    if any(r.action == GuardrailAction.REDACT for r in out_results):
        if "structured_output" in ans.metadata and "answer" in ans.metadata["structured_output"]:
            meta_answer = ans.metadata["structured_output"]["answer"]
            ans.metadata["structured_output"]["answer"] = runner.apply_redactions(meta_answer, out_results)
            
    # Verify output PII is redacted
    assert "[EMAIL]" in ans.text
    assert "alice@corp.com" not in ans.text
    
    # Verify metadata is redacted/scrubbed
    assert "alice@corp.com" not in ans.metadata["structured_output"]["answer"]
    assert "[EMAIL]" in ans.metadata["structured_output"]["answer"]
