"""Remote vision model adapter for the platform's OpenAI-compatible vLLM API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.models.oauth import OAuthError, OAuthTokenProvider
from golden_dataset_harness.models.provider_settings import ProviderSettings
from golden_dataset_harness.schemas.annotation import GroundingResult, QualityJudgment

logger = logging.getLogger(__name__)


class VLMServiceError(RuntimeError):
    """The remote vLLM service failed or returned malformed output."""


class OpenAICompatibleVLM(BaseVisionLanguageModel):
    """Vision adapter with OAuth refresh, structured output and bounded retries."""

    def __init__(
        self,
        settings: ProviderSettings | None = None,
        token_provider: OAuthTokenProvider | None = None,
        http_client: httpx.AsyncClient | None = None,
        **_: Any,
    ) -> None:
        self.settings = settings or ProviderSettings()
        super().__init__(
            api_base_url=self.settings.openai_base_url,
            model_name=self.settings.vllm_model,
            timeout=int(self.settings.vllm_timeout_seconds),
            max_retries=self.settings.vllm_max_retries,
        )
        self.token_provider = token_provider or OAuthTokenProvider(self.settings)
        self._http_client = http_client

    @property
    def model_id(self) -> str:
        return self.settings.vllm_model

    def _encode_image(self, image: bytes) -> str:
        """Normalize and shrink images so the Base64 message stays below the API limit."""
        try:
            source = Image.open(io.BytesIO(image)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Input is not a supported image") from exc

        source.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        quality = 88
        while True:
            buffer = io.BytesIO()
            source.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded_bytes = buffer.getvalue()
            if len(encoded_bytes) <= self.settings.vllm_max_image_bytes:
                return base64.b64encode(encoded_bytes).decode("ascii")
            if quality > 50:
                quality -= 10
                continue
            width, height = source.size
            if max(width, height) <= 224:
                raise ValueError("Image cannot be compressed below the API content limit")
            source = source.resize(
                (max(1, int(width * 0.8)), max(1, int(height * 0.8))),
                Image.Resampling.LANCZOS,
            )

    def _prepare_messages(self, image: bytes, text: str) -> list[dict[str, Any]]:
        encoded = self._encode_image(image)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                }},
            ],
        }]

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self.settings.vllm_enable_thinking}
        }
        if self.settings.vllm_guardrail != "off":
            body.update({
                "guardrail": self.settings.vllm_guardrail,
                "guard_output_mode": self.settings.vllm_guard_output_mode,
            })
        return body

    @staticmethod
    def _json_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {"type": "json_schema", "json_schema": {
            "name": name,
            "schema": schema,
        }}

    async def _call_api(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        last_error: Exception | None = None
        force_refresh = False
        for attempt in range(self.max_retries):
            owns_client = self._http_client is None
            client = self._http_client or httpx.AsyncClient(
                timeout=self.settings.vllm_timeout_seconds
            )
            try:
                token = await self.token_provider.get_token(force_refresh=force_refresh)
                force_refresh = False
                payload: dict[str, Any] = {
                    "model": self.settings.vllm_model,
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": self.settings.vllm_max_tokens,
                }
                # ``extra_body`` is an OpenAI SDK argument. With raw HTTP its
                # contents belong at the top level of the request JSON.
                payload.update(self._extra_body())
                if response_format is not None:
                    payload["response_format"] = response_format
                response = await client.post(
                    f"{self.settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code == 401 and attempt + 1 < self.max_retries:
                    await self.token_provider.invalidate()
                    force_refresh = True
                    continue
                if response.status_code >= 400:
                    if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                        raise VLMServiceError(
                            f"vLLM request failed with HTTP {response.status_code}"
                        )
                    last_error = httpx.HTTPStatusError(
                        "Transient vLLM error", request=response.request, response=response
                    )
                else:
                    try:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise VLMServiceError(
                            "vLLM returned an invalid Chat Completions response"
                        ) from exc
                    if not isinstance(content, str) or not content:
                        raise VLMServiceError("vLLM returned empty content")
                    return content
            except (httpx.RequestError, OAuthError) as exc:
                last_error = exc
            finally:
                if owns_client:
                    await client.aclose()
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(2**attempt)
        raise VLMServiceError("vLLM request failed after retries") from last_error

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VLMServiceError("vLLM returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise VLMServiceError("vLLM structured output must be a JSON object")
        return value

    async def generate_caption(self, image: bytes, prompt: str | None = None) -> str:
        text = prompt or (
            "Describe the visible person factually and concisely. Focus on clothing, colors, "
            "hair and accessories. Do not infer identity or hidden attributes."
        )
        return (await self._call_api(self._prepare_messages(image, text), 0.4)).strip()

    async def extract_attributes(
        self, image: bytes, taxonomy: dict[str, list[str]]
    ) -> dict[str, str]:
        properties = {
            key: {"type": "string", "enum": values}
            for key, values in taxonomy.items()
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        prompt = (
            "Extract only visually supported person attributes. For every field choose exactly "
            "one allowed enum. Use unknown when the attribute is hidden, ambiguous or absent."
        )
        content = await self._call_api(
            self._prepare_messages(image, prompt),
            0.0,
            self._json_schema("person_attributes", schema),
        )
        data = self._parse_json(content)
        for key, values in taxonomy.items():
            if data.get(key) not in values:
                raise VLMServiceError(f"Invalid value returned for attribute {key}")
        return {key: str(data[key]) for key in taxonomy}

    async def verify_claim(self, image: bytes, claim: str) -> GroundingResult:
        schema = {
            "type": "object",
            "properties": {
                "supported": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["supported", "confidence", "reason"],
            "additionalProperties": False,
        }
        prompt = (
            f"Verify this claim against visible evidence only: {claim!r}. "
            "If obscured or uncertain, supported must be false."
        )
        data = self._parse_json(await self._call_api(
            self._prepare_messages(image, prompt), 0.0,
            self._json_schema("grounding_result", schema),
        ))
        return GroundingResult(claim=claim, **data)

    async def judge_quality(
        self, image: bytes, caption: str, attributes: dict
    ) -> QualityJudgment:
        schema = {
            "type": "object",
            "properties": {
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["quality_score", "issues"],
            "additionalProperties": False,
        }
        prompt = (
            "Audit the annotation against the image for correctness, consistency, completeness "
            f"and hallucination. Caption: {caption!r}. Attributes: {json.dumps(attributes)}"
        )
        data = self._parse_json(await self._call_api(
            self._prepare_messages(image, prompt), 0.0,
            self._json_schema("quality_judgment", schema),
        ))
        return QualityJudgment(**data)
