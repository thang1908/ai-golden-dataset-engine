"""Direct extraction using the existing OAuth and structured-output adapter."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from golden_dataset_harness.baselines.contracts import InferenceResult
from golden_dataset_harness.baselines.io import (
    BaselineConfigError,
    check_output,
    json_hash,
    package_versions,
    run_images,
)
from golden_dataset_harness.models.oauth import OAuthTokenProvider
from golden_dataset_harness.models.openai_compatible import OpenAICompatibleVLM
from golden_dataset_harness.models.provider_settings import ProviderSettings
from golden_dataset_harness.schemas.taxonomy import TAXONOMY, attributes_json_schema


def load_settings(model_id: str | None) -> ProviderSettings:
    model_id = (model_id if model_id is not None else os.getenv("C1_VLLM_MODEL", "")).strip()
    if not model_id or model_id == "MODEL_ID_MEDIUM":
        raise BaselineConfigError("Set --model or C1_VLLM_MODEL to the service's Medium model ID")
    if model_id == "v-llm-v1-large":
        raise BaselineConfigError("C1 requires a Medium model ID, not v-llm-v1-large")
    try:
        return ProviderSettings(vllm_model=model_id)
    except ValidationError as exc:
        # ValidationError text can contain credentials from .env. Never expose it.
        raise BaselineConfigError(
            "Invalid V-LLM configuration; check host and OAuth credentials in "
            "golden_dataset_harness/.env"
        ) from exc


class VLLMPredictor:
    method = "c1_vllm_medium"

    def __init__(self, model: OpenAICompatibleVLM):
        self.model = model
        self.model_id = model.model_id
        self.model_calls = 0
        self.http_attempts = 0
        settings = model.settings
        prompt = model.attribute_prompt(TAXONOMY)
        self.metadata = {
            "prompt_version": "siglip-attributes-v1",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_schema_sha256": json_hash(attributes_json_schema(TAXONOMY)),
            "temperature": 0.0,
            "max_tokens": settings.vllm_max_tokens,
            "max_image_bytes": settings.vllm_max_image_bytes,
            "timeout_seconds": settings.vllm_timeout_seconds,
            "max_http_attempts": settings.vllm_max_retries,
            "enable_thinking": settings.vllm_enable_thinking,
            "guardrail": settings.vllm_guardrail,
            "guard_output_mode": settings.vllm_guard_output_mode,
            "preprocessing": "adapter JPEG, max 1280px, compressed to max_image_bytes",
            "packages": package_versions(["httpx", "pydantic", "Pillow"]),
        }

    async def count_request(self, request: httpx.Request) -> None:
        if request.url.path.endswith("/chat/completions"):
            self.http_attempts += 1

    async def predict(self, image: bytes) -> InferenceResult:
        self.model_calls += 1
        attributes = await self.model.extract_attributes(image, TAXONOMY)
        return InferenceResult(attributes=attributes)


async def run(
    paths: list[Path], output_dir: Path, settings: ProviderSettings, *, overwrite: bool = False,
) -> dict:
    check_output(output_dir, overwrite)
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.vllm_timeout_seconds) as client:
        provider = OAuthTokenProvider(settings, client)
        model = OpenAICompatibleVLM(settings, provider, client)
        predictor = VLLMPredictor(model)
        client.event_hooks["request"].append(predictor.count_request)
        return await run_images(
            predictor, paths, output_dir, overwrite=overwrite,
            model_load_ms=(time.perf_counter() - start) * 1000, started_at=started_at,
        )
