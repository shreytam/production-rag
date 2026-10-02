from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
from pydantic import BaseModel

from core.types import ChatMessage
from providers.generators.openai_compatible import OpenAICompatibleGenerator


class _Out(BaseModel):
    answer: str


def _bad_request(msg):
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(400, request=req)
    return openai.BadRequestError(msg, response=resp, body=None)


def _completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="m",
    )


def _gen(side_effect):
    g = OpenAICompatibleGenerator("m", "http://x", "k")
    g._client = MagicMock()
    g._client.chat.completions.create.side_effect = side_effect
    return g, g._client.chat.completions.create


def test_context_length_bad_request_is_not_retried_as_json_object():
    err = _bad_request("This model's maximum context length is 8192 tokens")
    g, create = _gen([err])
    try:
        g.complete([ChatMessage(role="user", content="hi")], response_model=_Out)
    except openai.BadRequestError:
        pass
    else:
        raise AssertionError("expected BadRequestError to propagate")
    assert create.call_count == 1


def test_unsupported_response_format_falls_back_to_json_object():
    err = _bad_request("'response_format' of type 'json_schema' is not supported")
    g, create = _gen([err, _completion('{"answer": "x"}')])
    out = g.complete([ChatMessage(role="user", content="hi")], response_model=_Out)
    assert create.call_count == 2
    assert create.call_args_list[1].kwargs["response_format"] == {"type": "json_object"}
    assert out.parsed == {"answer": "x"}


def test_parse_retry_is_not_an_identical_call():
    g, create = _gen([_completion("not json"), _completion('{"answer": "x"}')])
    out = g.complete([ChatMessage(role="user", content="hi")], response_model=_Out)
    assert create.call_count == 2
    first, second = (c.kwargs["messages"] for c in create.call_args_list)
    assert second != first
    assert "json" in second[-1]["content"].lower()
    assert out.parsed == {"answer": "x"}


def test_parse_failure_twice_leaves_parsed_none():
    g, create = _gen([_completion("nope"), _completion("still nope")])
    out = g.complete([ChatMessage(role="user", content="hi")], response_model=_Out)
    assert create.call_count == 2
    assert out.parsed is None
