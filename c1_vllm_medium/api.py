"""One V-LLM vision operation: image in, validated caption and attributes out."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from c1_vllm_medium.auth import TokenProvider
from c1_vllm_medium.config import Settings
from c1_vllm_medium.errors import ApiError, ResponseValidationError
from c1_vllm_medium.image import to_data_url
from c1_vllm_medium.taxonomy import json_schema, validate_attributes

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def annotation_prompt() -> str:
    return (
        "Analyze the visible person in the image and return only the requested JSON. "
        "Write caption as one concise, factual sentence describing only visible appearance, "
        "clothing and carried items. Do not infer identity, relationships, location, intent "
        "or hidden details. Use the exact attribute codes in the JSON Schema. Return every "
        "attribute field. "
        "For a single-label field, choose one allowed code or null when the person or "
        "relevant region is not visible or cannot be determined. For a multi-label field, "
        "return every visible allowed code, [] only when the region is visible and none "
        "apply, or null when it cannot be determined. Do not guess identity. "
        "Use unknown and none only where the JSON Schema permits them."
    )


def annotation_json_schema() -> dict[str, Any]:
    """Schema for the single C1 model response."""
    return {
        "type": "object",
        "properties": {
            "caption": {"type": "string", "minLength": 1, "maxLength": 500},
            "attributes": json_schema(),
        },
        "required": ["caption", "attributes"],
        "additionalProperties": False,
    }


def request_body(settings: Settings, image_data_url: str) -> dict[str, Any]:
    """The documented OpenAI-compatible Chat Completions request body."""
    return {
        "model": settings.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": annotation_prompt()},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "stream": False,
        "temperature": 0,
        "max_tokens": settings.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person_annotation", "schema": annotation_json_schema()},
        },
    }


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return min(8.0, 0.5 * (2 ** attempt))


class VllmAttributeClient:
    """Synchronous client with token refresh and bounded transient retry."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = client is None
        self._tokens = TokenProvider(settings, self._client)
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VllmAttributeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def annotate(self, image: bytes) -> dict[str, Any]:
        body = request_body(self._settings, to_data_url(image))
        response: httpx.Response | None = None
        token_refresh_used = False
        for attempt in range(self._settings.max_retries):
            try:
                token = self._tokens.get(force_refresh=token_refresh_used)
                token_refresh_used = False
                response = self._client.post(
                    f"{self._settings.service_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            except httpx.HTTPError as exc:
                if attempt + 1 == self._settings.max_retries:
                    raise ApiError("Chat Completions request failed after retries") from exc
                self._sleep(_retry_delay(None, attempt))
                continue
            if response.status_code == 401 and attempt + 1 < self._settings.max_retries:
                self._tokens.invalidate()
                token_refresh_used = True
                continue
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt + 1 < self._settings.max_retries
            ):
                self._sleep(_retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise ApiError(f"Chat Completions request failed with HTTP {response.status_code}")
            return parse_annotation(response)
        raise ApiError("Chat Completions request failed after retries")


def parse_annotation(response: httpx.Response) -> dict[str, Any]:
    """Extract, validate and normalize the caption plus all 21 attributes."""
    try:
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"caption", "attributes"}:
            raise ValueError("response does not match the annotation contract")
        caption = value["caption"]
        if not isinstance(caption, str) or not (caption := caption.strip()) or len(caption) > 500:
            raise ValueError("caption is empty or too long")
        return {"caption": caption, "attributes": validate_attributes(value["attributes"])}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResponseValidationError("Model returned an invalid structured annotation") from exc
