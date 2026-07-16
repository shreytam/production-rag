"""Anthropic generator using the Anthropic Messages API with tool-forced structured output."""

from __future__ import annotations

import json
import logging

import anthropic
from pydantic import BaseModel

from core.types import ChatMessage, LLMResponse, Usage

logger = logging.getLogger(__name__)

_TOOL_NAME = "structured_output"


class AnthropicGenerator:
    """Generator that uses the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        # Separate system message from user/assistant messages
        system_text = ""
        anthropic_messages: list[dict] = []

        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            kwargs["system"] = system_text

        # Add temperature only if non-zero (Anthropic accepts it but 0.0 is fine)
        if temperature != 0.0:
            kwargs["temperature"] = temperature

        if response_model is not None:
            tool_def = {
                "name": _TOOL_NAME,
                "description": f"Return structured output matching {response_model.__name__}",
                "input_schema": response_model.model_json_schema(),
            }
            kwargs["tools"] = [tool_def]
            kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}

        response = self._client.messages.create(**kwargs)

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        parsed = None
        text = ""

        if response_model is not None:
            # Find the tool_use block
            for block in response.content:
                if block.type == "tool_use":
                    parsed = block.input
                    text = json.dumps(parsed)
                    break
            # If no tool_use block found, leave parsed=None
            if parsed is None:
                for block in response.content:
                    if block.type == "text":
                        text = block.text
                        break
        else:
            # Plain text response
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break

        return LLMResponse(
            text=text,
            parsed=parsed,
            usage=usage,
            model=response.model,
        )
