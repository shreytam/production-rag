"""Tests for LLM providers (embedders and generators). No network calls."""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from core.config import Settings
from core.types import ChatMessage


# ─── Tiny pydantic model for structured-output tests ────────────────────────

class AnswerSchema(BaseModel):
    answer: str
    confidence: float


# ─── Helpers for building fake SDK responses ────────────────────────────────

def _make_embedding_item(vector: list[float]) -> types.SimpleNamespace:
    return types.SimpleNamespace(embedding=vector)


def _make_embeddings_response(vectors: list[list[float]]) -> types.SimpleNamespace:
    return types.SimpleNamespace(data=[_make_embedding_item(v) for v in vectors])


def _make_openai_chat_response(
    content: str,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> types.SimpleNamespace:
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_anthropic_response(
    tool_input: dict,
    model: str = "claude-test",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> types.SimpleNamespace:
    tool_block = types.SimpleNamespace(type="tool_use", input=tool_input)
    usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return types.SimpleNamespace(content=[tool_block], usage=usage, model=model)


def _make_anthropic_text_response(
    text: str,
    model: str = "claude-test",
    input_tokens: int = 5,
    output_tokens: int = 10,
) -> types.SimpleNamespace:
    text_block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return types.SimpleNamespace(content=[text_block], usage=usage, model=model)


# ─── Embedder tests ──────────────────────────────────────────────────────────

class TestOpenAICompatibleEmbedder:
    def _make_settings(self, batch_size: int = 64) -> Settings:
        return Settings(
            embed_api_key="test-key",
            embed_batch_size=batch_size,
            embed_dimension=1024,
            embed_model="test-embed-model",
            nvidia_api_key="test-nvidia-key",
        )

    def test_embed_query_returns_list_of_floats(self):
        """embed_query must return a list of floats."""
        vector = [0.1] * 1024
        fake_response = _make_embeddings_response([vector])

        with patch("providers.embedders.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embeddings.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            embedder = OpenAICompatibleEmbedder(self._make_settings())

            result = embedder.embed_query("hello world")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_query_length_matches_dimension(self):
        """embed_query result length must match settings.embed_dimension."""
        vector = [0.5] * 1024
        fake_response = _make_embeddings_response([vector])

        with patch("providers.embedders.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embeddings.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            settings = self._make_settings()
            embedder = OpenAICompatibleEmbedder(settings)

            result = embedder.embed_query("test")

        assert len(result) == settings.embed_dimension

    def test_embed_documents_batches_correctly(self):
        """5 texts with batch_size=2 should trigger 3 separate create() calls."""
        # batch_size=2: batches of [2, 2, 1] → 3 calls
        def side_effect(model, input, **kwargs):
            vectors = [[float(i)] * 8 for i in range(len(input))]
            return _make_embeddings_response(vectors)

        with patch("providers.embedders.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embeddings.create.side_effect = side_effect
            MockClient.return_value = mock_instance

            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            settings = self._make_settings(batch_size=2)
            embedder = OpenAICompatibleEmbedder(settings)

            texts = ["a", "b", "c", "d", "e"]
            results = embedder.embed_documents(texts)

        assert mock_instance.embeddings.create.call_count == 3
        assert len(results) == 5

    def test_nim_input_type_routing(self):
        """Against a NVIDIA base_url, docs embed as 'passage' and queries as 'query'."""
        captured = []

        def side_effect(model, input, **kwargs):
            captured.append(kwargs.get("extra_body", {}).get("input_type"))
            return _make_embeddings_response([[0.1] * 8 for _ in input])

        with patch("providers.embedders.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embeddings.create.side_effect = side_effect
            MockClient.return_value = mock_instance

            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            embedder = OpenAICompatibleEmbedder(self._make_settings())
            embedder.embed_documents(["a", "b"])
            embedder.embed_query("q")

        assert captured == ["passage", "query"]

    def test_non_nvidia_base_url_omits_input_type(self):
        """A real-OpenAI base_url must NOT send the NIM-only input_type field."""
        captured = []

        def side_effect(model, input, **kwargs):
            captured.append("extra_body" in kwargs)
            return _make_embeddings_response([[0.1] * 8 for _ in input])

        with patch("providers.embedders.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embeddings.create.side_effect = side_effect
            MockClient.return_value = mock_instance

            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            settings = Settings(
                embed_api_key="k",
                embed_base_url="https://api.openai.com/v1",
                embed_model="text-embedding-3-large",
            )
            OpenAICompatibleEmbedder(settings).embed_query("q")

        assert captured == [False]

    def test_dimension_property(self):
        """dimension property must return settings.embed_dimension."""
        with patch("providers.embedders.openai_compatible.openai.OpenAI"):
            from providers.embedders.openai_compatible import OpenAICompatibleEmbedder
            settings = self._make_settings()
            embedder = OpenAICompatibleEmbedder(settings)
            assert embedder.dimension == 1024


# ─── OpenAI Generator tests ──────────────────────────────────────────────────

class TestOpenAICompatibleGenerator:
    def test_complete_with_response_model(self):
        """Generator with response_model should set .parsed and .usage.total_tokens."""
        payload = {"answer": "Paris", "confidence": 0.95}
        content = json.dumps(payload)
        fake_response = _make_openai_chat_response(
            content=content,
            model="test-model",
            prompt_tokens=15,
            completion_tokens=25,
        )

        with patch("providers.generators.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.generators.openai_compatible import OpenAICompatibleGenerator
            gen = OpenAICompatibleGenerator(
                model="test-model",
                base_url="https://api.test.com/v1",
                api_key="test-key",
            )

            messages = [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is the capital of France?"),
            ]
            result = gen.complete(messages, response_model=AnswerSchema)

        assert result.parsed is not None
        assert result.parsed["answer"] == "Paris"
        assert abs(result.parsed["confidence"] - 0.95) < 1e-6
        assert result.usage.total_tokens == 40
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.model == "test-model"

    def test_complete_without_response_model(self):
        """Generator without response_model should return plain text response."""
        fake_response = _make_openai_chat_response(
            content="Hello, world!",
            model="test-model",
            prompt_tokens=5,
            completion_tokens=10,
        )

        with patch("providers.generators.openai_compatible.openai.OpenAI") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.generators.openai_compatible import OpenAICompatibleGenerator
            gen = OpenAICompatibleGenerator(
                model="test-model",
                base_url="https://api.test.com/v1",
                api_key="test-key",
            )

            messages = [ChatMessage(role="user", content="Say hello")]
            result = gen.complete(messages)

        assert result.text == "Hello, world!"
        assert result.parsed is None
        assert result.usage.total_tokens == 15


# ─── Anthropic Generator tests ───────────────────────────────────────────────

class TestAnthropicGenerator:
    def test_complete_with_response_model_uses_tool_use(self):
        """AnthropicGenerator should return .parsed from the tool_use block."""
        tool_input = {"answer": "Berlin", "confidence": 0.88}
        fake_response = _make_anthropic_response(
            tool_input=tool_input,
            model="claude-test",
            input_tokens=12,
            output_tokens=18,
        )

        with patch("providers.generators.anthropic.anthropic.Anthropic") as MockClient:
            mock_instance = MagicMock()
            mock_instance.messages.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.generators.anthropic import AnthropicGenerator
            gen = AnthropicGenerator(model="claude-test", api_key="test-key")

            messages = [
                ChatMessage(role="system", content="You are a knowledgeable assistant."),
                ChatMessage(role="user", content="What is the capital of Germany?"),
            ]
            result = gen.complete(messages, response_model=AnswerSchema)

        assert result.parsed is not None
        assert result.parsed["answer"] == "Berlin"
        assert abs(result.parsed["confidence"] - 0.88) < 1e-6
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 18
        assert result.usage.total_tokens == 30
        assert result.model == "claude-test"

    def test_complete_without_response_model(self):
        """AnthropicGenerator without response_model should return plain text."""
        fake_response = _make_anthropic_text_response(
            text="The answer is 42.",
            model="claude-test",
            input_tokens=8,
            output_tokens=6,
        )

        with patch("providers.generators.anthropic.anthropic.Anthropic") as MockClient:
            mock_instance = MagicMock()
            mock_instance.messages.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.generators.anthropic import AnthropicGenerator
            gen = AnthropicGenerator(model="claude-test", api_key="test-key")

            messages = [ChatMessage(role="user", content="What is 6 * 7?")]
            result = gen.complete(messages)

        assert result.text == "The answer is 42."
        assert result.parsed is None
        assert result.usage.total_tokens == 14

    def test_system_message_extracted(self):
        """System messages should be passed as system= parameter, not in messages list."""
        fake_response = _make_anthropic_text_response("ok", model="claude-test")

        with patch("providers.generators.anthropic.anthropic.Anthropic") as MockClient:
            mock_instance = MagicMock()
            mock_instance.messages.create.return_value = fake_response
            MockClient.return_value = mock_instance

            from providers.generators.anthropic import AnthropicGenerator
            gen = AnthropicGenerator(model="claude-test", api_key="test-key")

            messages = [
                ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="Hi"),
            ]
            gen.complete(messages)

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "Be concise."
        # Only the user message should be in messages list
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"


# ─── Protocol conformance checks ─────────────────────────────────────────────

def test_embedder_implements_protocol():
    """OpenAICompatibleEmbedder must satisfy the Embedder protocol."""
    from core.interfaces import Embedder
    from providers.embedders.openai_compatible import OpenAICompatibleEmbedder

    with patch("providers.embedders.openai_compatible.openai.OpenAI"):
        settings = Settings(embed_api_key="x", nvidia_api_key="x")
        embedder = OpenAICompatibleEmbedder(settings)

    assert isinstance(embedder, Embedder)


def test_openai_generator_implements_protocol():
    """OpenAICompatibleGenerator must satisfy the Generator protocol."""
    from core.interfaces import Generator
    from providers.generators.openai_compatible import OpenAICompatibleGenerator

    with patch("providers.generators.openai_compatible.openai.OpenAI"):
        gen = OpenAICompatibleGenerator("model", "http://base", "key")

    assert isinstance(gen, Generator)


def test_anthropic_generator_implements_protocol():
    """AnthropicGenerator must satisfy the Generator protocol."""
    from core.interfaces import Generator
    from providers.generators.anthropic import AnthropicGenerator

    with patch("providers.generators.anthropic.anthropic.Anthropic"):
        gen = AnthropicGenerator("model", "key")

    assert isinstance(gen, Generator)
