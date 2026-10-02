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
        seed: int | None = None,
    ) -> LLMResponse:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

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
        used_kwargs = kwargs
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Fall back to json_object ONLY when the 400 says the response_format /
            # json_schema / strict mode itself is unsupported. Any other
            # BadRequest (e.g. context length) would fail identically — re-raise.
            msg = str(exc).lower()
            schema_rejected = isinstance(exc, openai.BadRequestError) and any(
                tok in msg for tok in ("json_schema", "response_format", "strict", "schema")
            )
            if response_model is not None and schema_rejected:
                logger.warning("structured json_schema rejected; falling back to json_object: %s", exc)
                used_kwargs = dict(kwargs)
                used_kwargs["response_format"] = {"type": "json_object"}
                schema_str = json.dumps(response_model.model_json_schema())
                # Inject a system message to guide JSON output
                used_kwargs["messages"] = [
                    {
                        "role": "system",
                        "content": f"Respond with valid JSON matching this schema: {schema_str}",
                    },
                    *openai_messages,
                ]
                response = self._client.chat.completions.create(**used_kwargs)
            else:
                raise

        content = response.choices[0].message.content or ""
        usage_obj = response.usage
        prompt_tokens = usage_obj.prompt_tokens
        completion_tokens = usage_obj.completion_tokens

        parsed = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(content).model_dump()
            except Exception as parse_exc:
                # An identical temperature-0 call would just repeat the failure,
                # so retry ONCE with an explicit "valid JSON only" nudge.
                logger.warning("Parse failed on attempt 1; retrying with JSON nudge: %s", parse_exc)
                nudged = dict(used_kwargs)
                nudged["messages"] = [
                    *used_kwargs["messages"],
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Your reply was not valid JSON for the required schema. "
                        "Return ONLY valid JSON matching the schema, with no other text.",
                    },
                ]
                try:
                    retry_resp = self._client.chat.completions.create(**nudged)
                    retry_content = retry_resp.choices[0].message.content or ""
                    prompt_tokens += retry_resp.usage.prompt_tokens
                    completion_tokens += retry_resp.usage.completion_tokens
                    content = retry_content
                    parsed = response_model.model_validate_json(content).model_dump()
                except Exception as retry_exc:
                    logger.error("Parse failed on attempt 2; leaving parsed=None: %s", retry_exc)
                    parsed = None

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return LLMResponse(
            text=content,
            parsed=parsed,
            usage=usage,
            model=response.model,
        )
