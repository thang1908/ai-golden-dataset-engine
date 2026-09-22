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
from .prompts import annotation_prompt
from .taxonomy import attribute_json_schema, validate_attributes

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def annotation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "caption": {"type": "string", "minLength": 1, "maxLength": 500},
            "attributes": attribute_json_schema(),
        },
        "required": ["caption", "attributes"],
        "additionalProperties": False,
    }


def request_body(settings: Settings, image_data_url: str) -> dict[str, Any]:
    return {
        "model": settings.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": annotation_prompt()},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": settings.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person_annotation", "schema": annotation_json_schema()},
        },
    }


def _delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None and response.headers.get("retry-after"):
        try:
            return min(30.0, max(0.0, float(response.headers["retry-after"])))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2**attempt))


def parse_annotation(response: httpx.Response) -> dict[str, Any]:
    try:
        content = response.json()["choices"][0]["message"]["content"]
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"caption", "attributes"}:
            raise ValueError
        caption = value["caption"].strip()
        if not isinstance(value["caption"], str) or not caption or len(caption) > 500:
            raise ValueError
        return {"caption": caption, "attributes": validate_attributes(value["attributes"])}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResponseValidationError("Model returned an invalid structured annotation") from exc


class VllmClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings, self._client, self._owns_client, self._sleep = (
            settings,
            client or httpx.Client(timeout=settings.timeout_seconds),
            client is None,
            sleep,
        )
        self._tokens = TokenProvider(settings, self._client)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VllmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def annotate(self, image: bytes) -> dict[str, Any]:
        response: httpx.Response | None = None
        refresh = False
        body = request_body(self.settings, to_data_url(image))
        for attempt in range(self.settings.max_retries):
            try:
                token = self._tokens.get(force_refresh=refresh)
                refresh = False
                response = self._client.post(
                    f"{self.settings.service_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
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
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt + 1 < self.settings.max_retries
            ):
                self._sleep(_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise ApiError(f"Chat Completions request failed with HTTP {response.status_code}")
            return parse_annotation(response)
        raise ApiError("Chat Completions request failed after retries")

    def translate_caption(self, caption: str) -> str:
        body = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Translate this factual person-image caption into natural Vietnamese. Preserve only stated facts. Return only JSON.\nEnglish caption:\n" + caption}]}],
            "stream": False,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "caption_vietnamese", "schema": {"type": "object", "properties": {"caption_vi": {"type": "string", "minLength": 1, "maxLength": 500}}, "required": ["caption_vi"], "additionalProperties": False}}},
        }
        response: httpx.Response | None = None
        refresh = False
        for attempt in range(self.settings.max_retries):
            try:
                token = self._tokens.get(force_refresh=refresh)
                refresh = False
                response = self._client.post(f"{self.settings.service_url}/v1/chat/completions", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body)
            except httpx.HTTPError as exc:
                if attempt + 1 == self.settings.max_retries:
                    raise ApiError("Chat Completions request failed after retries") from exc
                self._sleep(_delay(None, attempt)); continue
            if response.status_code == 401 and attempt + 1 < self.settings.max_retries:
                self._tokens.invalidate(); refresh = True; continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < self.settings.max_retries:
                self._sleep(_delay(response, attempt)); continue
            if response.status_code >= 400:
                raise ApiError(f"Chat Completions request failed with HTTP {response.status_code}")
            try:
                value = json.loads(response.json()["choices"][0]["message"]["content"])
                caption_vi = value.get("caption_vi") if isinstance(value, dict) and set(value) == {"caption_vi"} else None
                if not isinstance(caption_vi, str) or not (caption_vi := caption_vi.strip()) or len(caption_vi) > 500:
                    raise ValueError
                return caption_vi
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ResponseValidationError("Model returned an invalid Vietnamese caption") from exc
        raise ApiError("Chat Completions request failed after retries")
