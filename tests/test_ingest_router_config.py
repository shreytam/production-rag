from core.config import Settings


def test_router_fills_chat_roles_but_not_embedder_or_reranker():
    s = Settings(
        llm_base_url="https://router.example/v1",
        llm_api_key="router-key",
        nvidia_api_key="",  # ensure no other fallback masks the router
    )
    # Chat roles follow the OpenAI-compatible router...
    for url in (s.gen_base_url, s.context_base_url, s.judge_base_url):
        assert url == "https://router.example/v1"
    for key in (s.gen_api_key, s.context_api_key, s.judge_api_key):
        assert key == "router-key"
    # ...but embeddings do NOT: they must stay coherent with the vector index
    # they built (model + dimension), and aggregators may not serve the same
    # embedding model. Embeddings keep their own base_url/key knobs.
    assert s.embed_base_url != "https://router.example/v1"
    assert s.embed_base_url == "https://integrate.api.nvidia.com/v1"
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
    # Embeddings are never dragged to the chat router.
    assert s.embed_base_url == "https://integrate.api.nvidia.com/v1"


def test_openrouter_key_used_for_openrouter_base_urls():
    s = Settings(
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key="",  # pin: neutralise ambient infra/.env
        openrouter_api_key="sk-or-test",
        nvidia_api_key="nv-key",
    )
    assert s.gen_api_key == "sk-or-test"
    # Embeddings still resolve to the NVIDIA key even with a router present.
    assert s.embed_api_key == "nv-key"
