"""Tests for VlmClient (with a stubbed OpenAI client)."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.vcc_common import CoTStructuredOutput
from app.vlm_client import VlmClient


def _make_client(response_text=None, raise_exc=None):
    client = VlmClient(base_url="http://stub:1234", model="stub-model", timeout=1)
    fake_openai = MagicMock()
    if raise_exc is not None:
        fake_openai.chat.completions.create.side_effect = raise_exc
    else:
        fake_openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )
    client.client = fake_openai
    return client, fake_openai


def test_base_url_normalized_with_v1():
    client = VlmClient(base_url="http://stub:1234")
    assert client.base_url.endswith("/v1")


def test_max_retries_disabled():
    """No SDK-level retries: callers need a predictable, bounded worst-case
    latency per VLM call rather than up to 3x the timeout hidden inside the
    openai client."""
    client = VlmClient(base_url="http://stub:1234")
    assert client.client.max_retries == 0


def test_default_timeout_from_env(monkeypatch):
    monkeypatch.setenv("LLAMA_REQUEST_TIMEOUT", "7")
    client = VlmClient(base_url="http://stub:1234")
    assert client.timeout == 7.0
    assert client.client.timeout == 7.0


def test_default_timeout_falls_back_to_five_seconds(monkeypatch):
    monkeypatch.delenv("LLAMA_REQUEST_TIMEOUT", raising=False)
    client = VlmClient(base_url="http://stub:1234")
    assert client.timeout == 5.0


def test_explicit_timeout_overrides_env(monkeypatch):
    monkeypatch.setenv("LLAMA_REQUEST_TIMEOUT", "7")
    client = VlmClient(base_url="http://stub:1234", timeout=1.5)
    assert client.timeout == 1.5


def test_query_structured_returns_parsed_schema():
    body = json.dumps({"description": "a person holding a gun", "final_answer": True})
    client, _ = _make_client(response_text=body)
    result = client.query_structured(
        "data:image/jpeg;base64,AAAA", "prompt", CoTStructuredOutput, low_res=True
    )
    assert result == CoTStructuredOutput(description="a person holding a gun", final_answer=True)


def test_query_structured_sends_expected_message_and_response_format():
    body = json.dumps({"description": "nothing", "final_answer": False})
    client, fake = _make_client(response_text=body)
    client.query_structured(
        "data:image/jpeg;base64,AAAA", "the prompt", CoTStructuredOutput, low_res=False
    )
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "stub-model"
    message = kwargs["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "the prompt"}
    assert message["content"][1]["image_url"]["detail"] == "high"
    assert message["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "CoTStructuredOutput"
    assert response_format["json_schema"]["schema"] == CoTStructuredOutput.model_json_schema()


def test_query_structured_returns_none_on_exception():
    client, _ = _make_client(raise_exc=RuntimeError("boom"))
    result = client.query_structured(
        "data:image/jpeg;base64,AAAA", "p", CoTStructuredOutput
    )
    assert result is None


def test_query_structured_returns_none_on_malformed_response():
    client = VlmClient(base_url="http://stub:1234")
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(choices=[])
    client.client = fake
    result = client.query_structured(
        "data:image/jpeg;base64,AAAA", "p", CoTStructuredOutput
    )
    assert result is None


def test_query_structured_returns_none_on_schema_mismatch():
    client, _ = _make_client(response_text="not json at all")
    result = client.query_structured(
        "data:image/jpeg;base64,AAAA", "p", CoTStructuredOutput
    )
    assert result is None

