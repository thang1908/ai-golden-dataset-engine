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

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None and response.headers.get("retry-after"):
        try:
            return min(30.0, max(0.0, float(response.headers["retry-after"])))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2**attempt))


class VllmClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = client is None
        self._tokens = TokenProvider(settings, self._client)
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VllmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete(
        self, *, prompt: str, schema_name: str, schema: dict[str, Any], image: bytes | None = None
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
        body = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            # Keep translation/schema-only calls deterministic; vision/reasoning stages still use
            # the shared VLLM_ENABLE_THINKING setting.
            "chat_template_kwargs": {
                "enable_thinking": self.settings.enable_thinking and schema_name != "caption_vietnamese"
            },
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
        }
        response: httpx.Response | None = None
        refresh = False
        for attempt in range(self.settings.max_retries):
            try:
                token = self._tokens.get(force_refresh=refresh)
                refresh = False
                response = self._client.post(
                    f"{self.settings.service_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=body,
                )
            except httpx.HTTPError as exc:
                if attempt + 1 == self.settings.max_retries:
                    raise ApiError("Chat Completions request failed after retries") from exc
                self._sleep(_delay(None, attempt))
                continue
            if response.status_code == 401 and attempt + 1 < self.settings.max_retries:
                self._tokens.invalidate()
                refresh = True
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < self.settings.max_retries:
                self._sleep(_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise ApiError(f"Chat Completions request failed with HTTP {response.status_code}")
            try:
                value = json.loads(response.json()["choices"][0]["message"]["content"])
                if not isinstance(value, dict):
                    raise ValueError
                return value
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ResponseValidationError("Model returned invalid structured JSON") from exc
        raise ApiError("Chat Completions request failed after retries")
