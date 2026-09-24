from __future__ import annotations

import json
import mimetypes
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from .config import Settings
from .errors import GeminiError, ResponseValidationError


class GeminiJudge:
    """Thread-local official Gemini SDK adapter; no provider payload is logged."""

    def __init__(
        self,
        settings: Settings,
        generate: Callable[[str, bytes | None, str | None, dict[str, Any]], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._generate = generate
        self._sleep = sleep
        self._local = threading.local()

    @staticmethod
    def _error_code(error: Exception) -> int | None:
        for name in ("status_code", "code"):
            code = getattr(error, name, None)
            if isinstance(code, int):
                return code
        return None

    @classmethod
    def _is_retryable(cls, error: Exception) -> bool:
        code = cls._error_code(error)
        if code in {408, 429, 500, 502, 503, 504}:
            return True
        return type(error).__name__ in {
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "ConnectError",
            "ReadError",
            "WriteError",
            "ServerError",
        }

    @classmethod
    def _request_error(cls, error: Exception) -> GeminiError:
        code = cls._error_code(error)
        status = f", HTTP {code}" if code is not None else ""
        return GeminiError(f"Gemini evaluation request failed ({type(error).__name__}{status})")

    def _sdk_client(self):
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiError(
                "google-genai is not installed; install project dependencies"
            ) from exc
        client = genai.Client(
            api_key=self._settings.api_key,
            http_options=types.HttpOptions(
                timeout=int(self._settings.timeout_seconds * 1000),
            ),
        )
        self._local.client = client
        return client

    def complete(
        self,
        *,
        prompt: str,
        image: bytes | None,
        image_path_name: str | None,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        mime_type = (
            mimetypes.guess_type(image_path_name)[0] or "image/jpeg"
            if image is not None and image_path_name is not None
            else None
        )
        for retry_number in range(self._settings.max_retries + 1):
            try:
                if self._generate is not None:
                    text = self._generate(prompt, image, mime_type, response_schema)
                else:
                    client = self._sdk_client()
                    from google.genai import types

                    contents: list[Any] = [prompt]
                    if image is not None and mime_type is not None:
                        contents.insert(0, types.Part.from_bytes(data=image, mime_type=mime_type))
                    response = client.models.generate_content(
                        model=self._settings.model,
                        contents=contents,
                        config={
                            "temperature": 0,
                            "max_output_tokens": self._settings.max_tokens,
                            "response_mime_type": "application/json",
                            "response_json_schema": response_schema,
                            "automatic_function_calling": {"disable": True},
                        },
                    )
                    text = response.text
                break
            except GeminiError:
                raise
            except Exception as exc:
                if (
                    not self._is_retryable(exc)
                    or retry_number == self._settings.max_retries
                ):
                    raise self._request_error(exc) from exc
                delay_seconds = min(2**retry_number, 16) + random.uniform(0, 0.5)
                self._sleep(delay_seconds)
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ResponseValidationError("Gemini returned invalid structured JSON") from exc
        if not isinstance(value, dict):
            raise ResponseValidationError("Gemini returned a non-object structured response")
        return value
