"""OpenAI-compatible generator (works with NVIDIA NIM, OpenAI, or any compatible API)."""

from __future__ import annotations

import json
import logging

import openai
from pydantic import BaseModel

from core.types import ChatMessage, LLMResponse, Usage

logger = logging.getLogger(__name__)


class OpenAICompatibleGenerator:
    """Generator that uses the OpenAI chat completions API (or any compatible endpoint)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        # Generous timeout + automatic retries with exponential backoff: NIM can
        # be slow to respond under high traffic, and we'd rather wait than fail.
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_model is not None:
            schema = response_model.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

        # --- First attempt ---
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Fallback: a model that rejects strict json_schema typically 400s with
            # a message about response_format / json_schema / strict / schema. We
            # trigger the json_object fallback on any BadRequest while a schema was
            # requested (the request shape was refused), not just a literal match.
            is_bad_request = isinstance(exc, openai.BadRequestError)
            mentions_schema = any(
                tok in str(exc).lower()
                for tok in ("json_schema", "response_format", "schema", "strict")
            )
            if response_model is not None and (is_bad_request or mentions_schema):
                logger.warning("structured json_schema rejected; falling back to json_object: %s", exc)
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["response_format"] = {"type": "json_object"}
                schema_str = json.dumps(response_model.model_json_schema())
                # Inject a system message to guide JSON output
                fallback_messages = list(openai_messages)
                system_instruction = {
                    "role": "system",
                    "content": f"Respond with valid JSON matching this schema: {schema_str}",
                }
                fallback_messages.insert(0, system_instruction)
                fallback_kwargs["messages"] = fallback_messages
                response = self._client.chat.completions.create(**fallback_kwargs)
            else:
                raise

        content = response.choices[0].message.content or ""
        usage_obj = response.usage
        usage = Usage(
            prompt_tokens=usage_obj.prompt_tokens,
            completion_tokens=usage_obj.completion_tokens,
            total_tokens=usage_obj.total_tokens,
        )

        parsed = None
        if response_model is not None:
            # Try to parse; retry once on failure
            for attempt in range(2):
                try:
                    parsed_model = response_model.model_validate_json(content)
                    parsed = parsed_model.model_dump()
                    break
                except Exception as parse_exc:
                    if attempt == 0:
                        logger.warning("Parse failed on attempt 1; retrying: %s", parse_exc)
                        # retry: re-fetch with same kwargs
                        try:
                            retry_resp = self._client.chat.completions.create(**kwargs)
                            content = retry_resp.choices[0].message.content or ""
                        except Exception:
                            pass  # leave content as-is
                    else:
                        logger.error("Parse failed on attempt 2; leaving parsed=None: %s", parse_exc)
                        parsed = None

        return LLMResponse(
            text=content,
            parsed=parsed,
            usage=usage,
            model=response.model,
        )
