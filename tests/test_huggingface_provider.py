import httpx

from automatic_annotations.models import ModelSettings
from automatic_annotations.providers import HuggingFaceProvider


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def test_huggingface_uses_router_chat_completions_and_session_token(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "model": "MiniMaxAI/MiniMax-M2.7:novita",
                "choices": [{"message": {"content": '{"value":"ok"}'}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    settings = ModelSettings(
        provider="huggingface",
        model="MiniMaxAI/MiniMax-M2.7:novita",
        api_base="https://router.huggingface.co/hf-inference/models",
        api_key_env="OPENAI_API_KEY",
    )
    result = HuggingFaceProvider().extract("text", SCHEMA, "", [], settings, api_key="hf_session")
    assert result.error is None
    url, request = requests[0]
    assert url == "https://router.huggingface.co/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer hf_session"
    assert request["json"]["model"] == "MiniMaxAI/MiniMax-M2.7:novita"
    assert request["json"]["response_format"]["type"] == "json_schema"
    assert request["json"]["reasoning_effort"] == "low"


def test_huggingface_prefers_hf_token_over_stale_openai_env(monkeypatch):
    requests = []
    monkeypatch.setenv("HF_TOKEN", "hf_environment")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong_key")

    def post(url, **kwargs):
        requests.append(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    settings = ModelSettings(
        provider="huggingface",
        model="model:fastest",
        api_base="https://router.huggingface.co/v1",
        api_key_env="OPENAI_API_KEY",
    )
    result = HuggingFaceProvider().extract("text", SCHEMA, "", [], settings)
    assert result.error is None
    assert requests[0]["headers"]["Authorization"] == "Bearer hf_environment"


def test_huggingface_error_includes_response_body(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(
            401,
            request=httpx.Request("POST", url),
            json={"error": {"message": "Invalid username or password"}},
        )

    monkeypatch.setattr(httpx, "post", post)
    settings = ModelSettings(provider="huggingface", model="model", api_base="https://router.huggingface.co/v1")
    result = HuggingFaceProvider().extract("text", SCHEMA, "", [], settings, api_key="bad")
    assert "Invalid username or password" in result.error


def test_huggingface_omits_reasoning_when_provider_rejects_it(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if len(requests) == 1:
            return httpx.Response(
                400,
                request=httpx.Request("POST", url),
                json={"error": {
                    "message": "Unsupported parameter: reasoning_effort",
                    "param": "reasoning_effort",
                }},
            )
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    settings = ModelSettings(
        provider="huggingface",
        model="model:novita",
        api_base="https://router.huggingface.co/v1",
        reasoning_effort="low",
    )
    result = HuggingFaceProvider().extract("text", SCHEMA, "", [], settings, api_key="hf_token")
    assert result.error is None
    assert requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in requests[1]


def test_huggingface_model_default_omits_reasoning_effort(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    settings = ModelSettings(
        provider="huggingface",
        model="model",
        api_base="https://router.huggingface.co/v1",
        reasoning_effort="default",
    )
    result = HuggingFaceProvider().extract("text", SCHEMA, "", [], settings, api_key="hf_token")
    assert result.error is None
    assert "reasoning_effort" not in requests[0]
