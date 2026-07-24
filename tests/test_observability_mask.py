from core.config import Settings
from observability.langfuse_tracing import Tracer

def test_tracer_mask_callback():
    # Construct a local tracer with enabled=True mock
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk",
        langfuse_secret_key="sk"
    )
    
    # We will verify that if Langfuse is loaded, or when we intercept observation calls,
    # the strings containing PII are replaced recursively.
    # To mock this, we will instantiate Tracer and assert mask behavior.
    tracer = Tracer(settings)
    assert tracer._enabled is True
    
    # Define a helper callback like the one tracer will register with Langfuse
    mask_fn = getattr(tracer, "_mask_data", None)
    assert mask_fn is not None
    
    # Test simple string
    assert mask_fn("My SSN is 000-12-3456.") == "My SSN is [SSN]."
    # Test dictionary nested payload
    payload = {
        "question": "Ask alice@corp.com",
        "nested": {
            "phone": "Call 555-123-4567"
        },
        "list": ["another bob@corp.com", 123]
    }
    cleaned = mask_fn(payload)
    assert cleaned["question"] == "Ask [EMAIL]"
    assert cleaned["nested"]["phone"] == "Call [PHONE]"
    assert cleaned["list"][0] == "another [EMAIL]"
    assert cleaned["list"][1] == 123
