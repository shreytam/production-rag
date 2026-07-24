from core.config import Settings
from core.types import ACLContext, ScoredChunk, Chunk, LLMResponse, Usage, Answer
from core.pipeline import RAGPipeline
from guardrails.runner import GuardrailRunner
from guardrails.pii_guard import PIIGuardrail

class FakeGenerator:
    def complete(self, messages, **kwargs):
        # returns normal answer mimicking RAG system
        return LLMResponse(
            text="Reach bob@corp.com or john@corp.com.",
            model="fake-gen",
            usage=Usage(prompt_tokens=10, completion_tokens=10)
        )

def test_full_query_pipeline_pii_redaction(tmp_path):
    log_file = tmp_path / "pii_audit.jsonl"
    settings = Settings(
        pii_mode="redact",
        pii_audit_log_path=str(log_file),
        pii_scan_output=True,
        langfuse_enabled=False
    )
    
    # Mock retriever
    class FakeRetriever:
        def retrieve(self, query):
            c = Chunk(chunk_id="ch1", doc_id="d1", text="dummy", tenant_id="public")
            return [ScoredChunk(chunk=c, score=0.9)]
            
    class FakeGrounded:
        def generate(self, question, scored):
            # simulate structured metadata output
            ans_text = "The answer is help@site.com."
            return Answer(
                text=ans_text,
                usage=Usage(prompt_tokens=10, completion_tokens=10),
                metadata={
                    "structured_output": {
                        "answer": ans_text
                    }
                }
            )

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        grounded=FakeGrounded(),
        settings=settings,
        guardrails=GuardrailRunner(input_guards=[], output_guards=[PIIGuardrail()])
    )
    
    # 1. Run pipeline for query containing PII
    result = pipeline.run("What about help@site.com?", acl=ACLContext(tenant_id="public"))
    
    # Clean answer returned on normal run flow
    assert "[EMAIL]" in result["answer"]
    assert "help@site.com" not in result["answer"]
    
    # Verified metadata copy is also redacted
    answer_obj = result["answer_obj"]
    assert "help@site.com" not in answer_obj.metadata["structured_output"]["answer"]
    assert "[EMAIL]" in answer_obj.metadata["structured_output"]["answer"]
    
    # Guardrail logs contain no raw PII
    guard_logs = answer_obj.metadata["guardrails"]["output"]
    assert len(guard_logs) > 0
    # Findings array has no value block
    for finding in guard_logs[0]["metadata"]["findings"]:
        assert "value" not in finding
        assert "help@site.com" not in str(finding)
