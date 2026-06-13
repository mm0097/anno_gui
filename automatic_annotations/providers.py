from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import ExtractionResponse, ModelSettings, Usage
from .schema import build_prompt, openai_strict_schema, parse_json_object, validation_errors


class ExtractionProvider(ABC):
    @abstractmethod
    def extract(
        self,
        text: str,
        schema: dict[str, Any],
        instructions: str,
        examples: list[dict[str, Any]],
        settings: ModelSettings,
        api_key: str | None = None,
    ) -> ExtractionResponse: ...


class OpenAICompatibleProvider(ExtractionProvider):
    @staticmethod
    def _error_details(response: httpx.Response) -> tuple[str, str | None]:
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                return str(error.get("message", response.text)), error.get("param")
        except (ValueError, TypeError):
            pass
        return response.text, None

    @staticmethod
    def _is_official_openai(api_base: str) -> bool:
        return (urlparse(api_base).hostname or "").lower() == "api.openai.com"

    def extract(
        self,
        text: str,
        schema: dict[str, Any],
        instructions: str,
        examples: list[dict[str, Any]],
        settings: ModelSettings,
        api_key: str | None = None,
    ) -> ExtractionResponse:
        if self._is_official_openai(settings.api_base):
            return self._extract_openai_responses(
                text, schema, instructions, examples, settings, api_key
            )
        key = api_key or os.getenv(settings.api_key_env)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": build_prompt(schema, instructions, examples)},
                {"role": "user", "content": text},
            ],
            "temperature": settings.temperature,
            "max_completion_tokens": settings.max_output_tokens,
        }
        # OpenAI and many compatible servers support JSON Schema structured output.
        strict_schema = openai_strict_schema(schema)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "annotation", "strict": True, "schema": strict_schema},
        }
        started = time.perf_counter()
        try:
            url = f"{settings.api_base.rstrip('/')}/chat/completions"
            for _ in range(4):
                response = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=settings.timeout_seconds,
                )
                if response.status_code < 400:
                    break
                message, parameter = self._error_details(response)
                changed = False
                if parameter == "max_completion_tokens" or (
                    "max_completion_tokens" in message and "max_tokens" in message
                ):
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                    changed = True
                if parameter == "response_format" or "response_format" in message:
                    if payload.get("response_format", {}).get("type") == "json_schema":
                        payload["response_format"] = {"type": "json_object"}
                        changed = True
                if parameter == "temperature" or (
                    "temperature" in message and "default" in message.lower()
                ):
                    if "temperature" in payload:
                        payload.pop("temperature")
                        changed = True
                if not changed:
                    break
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            parsed = parse_json_object(raw)
            validation_schema = strict_schema if payload["response_format"]["type"] == "json_schema" else schema
            errors = validation_errors(parsed, validation_schema)
            usage = body.get("usage", {})
            return ExtractionResponse(
                parsed=parsed,
                raw_output=raw,
                usage=Usage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                ),
                latency_seconds=time.perf_counter() - started,
                model=body.get("model", settings.model),
                validation_errors=errors,
            )
        except Exception as exc:
            raw = response.text if "response" in locals() else ""
            return ExtractionResponse(
                raw_output=raw,
                latency_seconds=time.perf_counter() - started,
                model=settings.model,
                error=str(exc),
            )

    def _extract_openai_responses(
        self,
        text: str,
        schema: dict[str, Any],
        instructions: str,
        examples: list[dict[str, Any]],
        settings: ModelSettings,
        api_key: str | None,
    ) -> ExtractionResponse:
        key = api_key or os.getenv(settings.api_key_env)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        strict_schema = openai_strict_schema(schema)
        payload: dict[str, Any] = {
            "model": settings.model,
            "input": [
                {
                    "role": "system",
                    "content": build_prompt(
                        schema, instructions, examples, include_schema=False
                    ),
                },
                {"role": "user", "content": text},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "annotation",
                    "strict": True,
                    "schema": strict_schema,
                }
            },
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "store": False,
        }
        if settings.reasoning_effort != "default":
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        started = time.perf_counter()
        response = None
        try:
            url = f"{settings.api_base.rstrip('/')}/responses"
            for _ in range(3):
                response = httpx.post(
                    url, headers=headers, json=payload, timeout=settings.timeout_seconds
                )
                if response.status_code < 400:
                    break
                message, parameter = self._error_details(response)
                if parameter == "temperature" or (
                    "temperature" in message and "default" in message.lower()
                ):
                    payload.pop("temperature", None)
                    continue
                if parameter in {"reasoning", "reasoning.effort"} or (
                    "reasoning" in message.lower() and "support" in message.lower()
                ):
                    payload.pop("reasoning", None)
                    continue
                break
            response.raise_for_status()
            body = response.json()
            if body.get("status") == "incomplete":
                reason = (body.get("incomplete_details") or {}).get("reason", "unknown")
                raise ValueError(f"Incomplete OpenAI response: {reason}")
            raw = ""
            refusal = ""
            for output in body.get("output", []):
                if output.get("type") != "message":
                    continue
                for content in output.get("content", []):
                    if content.get("type") == "output_text":
                        raw += content.get("text", "")
                    elif content.get("type") == "refusal":
                        refusal += content.get("refusal", "")
            if refusal:
                raise ValueError(f"OpenAI refused the extraction: {refusal}")
            if not raw:
                raise ValueError("OpenAI response contained no structured output")
            parsed = parse_json_object(raw)
            usage = body.get("usage", {})
            return ExtractionResponse(
                parsed=parsed,
                raw_output=raw,
                usage=Usage(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                ),
                latency_seconds=time.perf_counter() - started,
                model=body.get("model", settings.model),
                validation_errors=validation_errors(parsed, strict_schema),
            )
        except Exception as exc:
            return ExtractionResponse(
                raw_output=response.text if response is not None else "",
                latency_seconds=time.perf_counter() - started,
                model=settings.model,
                error=str(exc),
            )


class HuggingFaceProvider(ExtractionProvider):
    @staticmethod
    def _chat_completions_url(api_base: str) -> str:
        base = api_base.strip().rstrip("/")
        parsed = urlparse(base)
        if parsed.hostname == "router.huggingface.co" and "/hf-inference/models" in parsed.path:
            return "https://router.huggingface.co/v1/chat/completions"
        if not base:
            return "https://router.huggingface.co/v1/chat/completions"
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def extract(
        self,
        text: str,
        schema: dict[str, Any],
        instructions: str,
        examples: list[dict[str, Any]],
        settings: ModelSettings,
        api_key: str | None = None,
    ) -> ExtractionResponse:
        key = api_key or os.getenv("HF_TOKEN") or os.getenv(settings.api_key_env)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self._chat_completions_url(settings.api_base)
        strict_schema = openai_strict_schema(schema)
        payload: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_prompt(schema, instructions, examples, include_schema=False),
                },
                {"role": "user", "content": text},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "annotation",
                    "schema": strict_schema,
                    "strict": True,
                },
            },
        }
        if settings.reasoning_effort != "default":
            payload["reasoning_effort"] = settings.reasoning_effort
        started = time.perf_counter()
        try:
            for _ in range(2):
                response = httpx.post(
                    url, headers=headers, json=payload, timeout=settings.timeout_seconds
                )
                if response.status_code < 400:
                    break
                message, parameter = OpenAICompatibleProvider._error_details(response)
                if parameter == "reasoning_effort" or (
                    "reasoning_effort" in message.lower()
                    and any(word in message.lower() for word in ("unsupported", "support", "unknown"))
                ):
                    payload.pop("reasoning_effort", None)
                    continue
                break
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            parsed = parse_json_object(raw)
            usage = body.get("usage", {})
            return ExtractionResponse(
                parsed=parsed,
                raw_output=raw,
                usage=Usage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                ),
                latency_seconds=time.perf_counter() - started,
                model=body.get("model", settings.model),
                validation_errors=validation_errors(parsed, strict_schema),
            )
        except Exception as exc:
            raw = response.text if "response" in locals() else ""
            detail = f"{exc}: {raw}" if raw else str(exc)
            return ExtractionResponse(
                raw_output=raw,
                latency_seconds=time.perf_counter() - started,
                model=settings.model,
                error=detail,
            )


def get_provider(name: str) -> ExtractionProvider:
    if name == "huggingface":
        return HuggingFaceProvider()
    if name == "openai_compatible":
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown provider: {name}")
