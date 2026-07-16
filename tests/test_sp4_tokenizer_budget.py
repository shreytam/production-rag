from core.context_assembly import resolve_encoding

def test_resolve_tokenizer_encoding():
    assert resolve_encoding("cl100k_base", "llama-model") == "cl100k_base"
    assert resolve_encoding("auto", "gpt-4o") == "o200k_base"
    assert resolve_encoding("auto", "llama-3-model") == "cl100k_base"
