from core.config import Settings


def test_router_fills_all_model_roles_but_not_reranker():
    s = Settings(
        llm_base_url="https://router.example/v1",
        llm_api_key="router-key",
        nvidia_api_key="",  # ensure no other fallback masks the router
    )
    for url in (s.embed_base_url, s.gen_base_url, s.context_base_url, s.judge_base_url):
        assert url == "https://router.example/v1"
    for key in (s.embed_api_key, s.gen_api_key, s.context_api_key, s.judge_api_key):
        assert key == "router-key"
    # Reranker is NOT routed through the OpenAI-compatible router.
    assert s.reranker_nim_base_url != "https://router.example/v1"


def test_explicit_role_override_beats_router():
    s = Settings(
        llm_base_url="https://router.example/v1",
        llm_api_key="router-key",
        gen_base_url="https://custom.example/v1",
        gen_api_key="custom-key",
    )
    assert s.gen_base_url == "https://custom.example/v1"
    assert s.gen_api_key == "custom-key"
    assert s.embed_base_url == "https://router.example/v1"
