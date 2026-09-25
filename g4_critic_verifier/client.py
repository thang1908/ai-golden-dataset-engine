from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from .auth import TokenProvider
from .config import Settings
from .errors import ApiError, ResponseValidationError
from .image import to_data_url
from .telemetry import CallContext, TelemetryWriter

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None and response.headers.get("retry-after"):
        try:
            return min(30.0, max(0.0, float(response.headers["retry-after"])))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2**attempt))


def _usage(payload: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None, None, None
    usage = payload["usage"]
    def value(name: str) -> int | None:
        raw = usage.get(name)
        return int(raw) if isinstance(raw, (int, float)) else None
    return value("prompt_tokens"), value("completion_tokens"), value("total_tokens")


class VllmClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        telemetry: TelemetryWriter | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = client is None
        self._tokens = TokenProvider(settings, self._client)
        self._sleep = sleep
        self._telemetry = telemetry

    def _record(self, context: CallContext | None, *, attempt: int, has_image: bool, outcome: str, latency_ms: float, http_status: int | None = None, payload: Any = None, retry_reason: str | None = None) -> None:
        if self._telemetry is None or context is None:
            return
        prompt_tokens, completion_tokens, total_tokens = _usage(payload)
        self._telemetry.record(context, http_attempt=attempt, model_id=self.settings.model, has_image=has_image, outcome=outcome, latency_ms=latency_ms, http_status=http_status, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, retry_reason=retry_reason)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VllmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete(
        self, *, prompt: str, schema_name: str, schema: dict[str, Any], image: bytes | None = None, call_context: CallContext | None = None
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
        body = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": min(self.settings.max_tokens, 512)
            if schema_name in {"critic_issues", "verification"}
            else self.settings.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self.settings.enable_thinking and schema_name == "draft_annotation"
            },
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
        }
        response: httpx.Response | None = None
        refresh = False
        for attempt in range(self.settings.max_retries):
            attempt_number = attempt + 1
            started = time.perf_counter()
            try:
                token = self._tokens.get(force_refresh=refresh)
                refresh = False
                response = self._client.post(
                    f"{self.settings.service_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=body,
                )
            except httpx.HTTPError as exc:
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="transport_error", latency_ms=(time.perf_counter() - started) * 1000, retry_reason=type(exc).__name__)
                if attempt + 1 == self.settings.max_retries:
                    raise ApiError("Chat Completions request failed after retries") from exc
                self._sleep(_delay(None, attempt))
                continue
            try:
                response_payload: Any = response.json()
            except (ValueError, TypeError):
                response_payload = None
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code == 401 and attempt + 1 < self.settings.max_retries:
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="http_error", latency_ms=latency_ms, http_status=401, payload=response_payload, retry_reason="http_401")
                self._tokens.invalidate()
                refresh = True
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < self.settings.max_retries:
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="http_error", latency_ms=latency_ms, http_status=response.status_code, payload=response_payload, retry_reason=f"http_{response.status_code}")
                self._sleep(_delay(response, attempt))
                continue
            if response.status_code >= 400:
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="http_error", latency_ms=latency_ms, http_status=response.status_code, payload=response_payload, retry_reason=f"http_{response.status_code}")
                raise ApiError(f"Chat Completions request failed with HTTP {response.status_code}")
            try:
                value = json.loads(response_payload["choices"][0]["message"]["content"])
                if not isinstance(value, dict):
                    raise ValueError
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="success", latency_ms=latency_ms, http_status=response.status_code, payload=response_payload)
                return value
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._record(call_context, attempt=attempt_number, has_image=image is not None, outcome="response_parse_error", latency_ms=latency_ms, http_status=response.status_code, payload=response_payload, retry_reason="invalid_structured_json")
                if attempt + 1 < self.settings.max_retries:
                    self._sleep(_delay(response, attempt))
                    continue
                raise ResponseValidationError(
                    f"{schema_name}: Model returned invalid structured JSON"
                ) from exc
        raise ApiError("Chat Completions request failed after retries")
