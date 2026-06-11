import httpx

from automatic_annotations.models import (
    EdgeTypeDefinition,
    GraphExtractionSchema,
    ModelSettings,
    NodeTypeDefinition,
)
from automatic_annotations.providers import OpenAICompatibleProvider
from automatic_annotations.schema import compile_graph_schema


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def success_response(model="test-model"):
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        json={
            "model": model,
            "choices": [{"message": {"content": '{"value":"ok"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )


def settings():
    return ModelSettings(model="test-model", api_base="https://example.test/v1", max_output_tokens=321)


def official_settings():
    return ModelSettings(model="gpt-5.4-mini", api_base="https://api.openai.com/v1", max_output_tokens=321)


def responses_success_response():
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        json={
            "id": "resp_test",
            "status": "completed",
            "model": "gpt-5.4-mini",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": '{"value":"ok"}'}],
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )


def test_uses_modern_completion_token_parameter(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        return success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], settings())
    assert result.error is None
    assert requests[0]["max_completion_tokens"] == 321
    assert "max_tokens" not in requests[0]


def test_falls_back_to_legacy_max_tokens_for_compatible_servers(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if len(requests) == 1:
            return httpx.Response(
                400,
                request=httpx.Request("POST", url),
                json={"error": {
                    "message": "Unsupported parameter: max_completion_tokens. Use max_tokens instead.",
                    "param": "max_completion_tokens",
                }},
            )
        return success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], settings())
    assert result.error is None
    assert requests[1]["max_tokens"] == 321
    assert "max_completion_tokens" not in requests[1]


def test_token_and_structured_output_fallbacks_can_both_apply(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if len(requests) == 1:
            return httpx.Response(
                400, request=httpx.Request("POST", url),
                json={"error": {"message": "Use max_tokens", "param": "max_completion_tokens"}},
            )
        if len(requests) == 2:
            return httpx.Response(
                400, request=httpx.Request("POST", url),
                json={"error": {"message": "response_format json_schema unsupported", "param": "response_format"}},
            )
        return success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], settings())
    assert result.error is None
    assert requests[2]["max_tokens"] == 321
    assert requests[2]["response_format"] == {"type": "json_object"}


def test_omits_temperature_when_model_only_supports_default(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if len(requests) == 1:
            return httpx.Response(
                400,
                request=httpx.Request("POST", url),
                json={"error": {
                    "message": "Unsupported value: temperature only supports the default (1) value.",
                    "param": "temperature",
                }},
            )
        return success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], settings())
    assert result.error is None
    assert requests[0]["temperature"] == 0.0
    assert "temperature" not in requests[1]


def test_all_openai_compatibility_fallbacks_can_apply(monkeypatch):
    requests = []
    errors = [
        {"message": "Use max_tokens", "param": "max_completion_tokens"},
        {"message": "Only the default temperature is supported", "param": "temperature"},
        {"message": "response_format json_schema unsupported", "param": "response_format"},
    ]

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if errors:
            return httpx.Response(
                400, request=httpx.Request("POST", url), json={"error": errors.pop(0)}
            )
        return success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], settings())
    assert result.error is None
    assert requests[-1]["max_tokens"] == 321
    assert "temperature" not in requests[-1]
    assert requests[-1]["response_format"] == {"type": "json_object"}


def test_official_openai_uses_responses_strict_structured_output(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return responses_success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "Be precise", [], official_settings())
    assert result.error is None
    url, payload = requests[0]
    assert url == "https://api.openai.com/v1/responses"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["store"] is False
    assert "JSON Schema" not in payload["input"][0]["content"]


def test_official_openai_does_not_fall_back_from_strict_schema(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(
            400,
            request=httpx.Request("POST", url),
            json={"error": {"message": "Invalid schema", "param": "text.format.schema"}},
        )

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], official_settings())
    assert result.error is not None
    assert "400" in result.error


def test_official_openai_reports_incomplete_and_refused_responses(monkeypatch):
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            json={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No"}]}],
            },
        ),
    ]

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: responses.pop(0))
    provider = OpenAICompatibleProvider()
    incomplete = provider.extract("text", SCHEMA, "", [], official_settings())
    refused = provider.extract("text", SCHEMA, "", [], official_settings())
    assert "Incomplete OpenAI response" in incomplete.error
    assert "refused" in refused.error


def test_official_openai_omits_reasoning_when_model_rejects_it(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        if len(requests) == 1:
            return httpx.Response(
                400,
                request=httpx.Request("POST", url),
                json={"error": {
                    "message": "This model does not support reasoning effort.",
                    "param": "reasoning.effort",
                }},
            )
        return responses_success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], official_settings())
    assert result.error is None
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert "reasoning" not in requests[1]


def test_official_openai_can_use_model_default_reasoning(monkeypatch):
    requests = []
    model_settings = official_settings().model_copy(update={"reasoning_effort": "default"})

    def post(url, **kwargs):
        requests.append(kwargs["json"].copy())
        return responses_success_response()

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", SCHEMA, "", [], model_settings)
    assert result.error is None
    assert "reasoning" not in requests[0]


def test_official_openai_request_contains_edge_endpoint_rules(monkeypatch):
    requests = []
    graph_schema = compile_graph_schema(GraphExtractionSchema(
        node_types=[NodeTypeDefinition(name="person"), NodeTypeDefinition(name="organization")],
        edge_types=[EdgeTypeDefinition(
            name="works_for", source_types=["person"], target_types=["organization"]
        )],
    ))

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "status": "completed",
                "model": "gpt-5.4-mini",
                "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": '{"nodes":[],"edges":[]}'
                }]}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    result = OpenAICompatibleProvider().extract("text", graph_schema, "", [], official_settings())
    assert result.error is None
    sent_schema = requests[0]["text"]["format"]["schema"]
    edge_variant = sent_schema["properties"]["edges"]["items"]["anyOf"][0]
    assert "Allowed source node types: person" in edge_variant["description"]
    assert "Allowed target node types: organization" in edge_variant["description"]
