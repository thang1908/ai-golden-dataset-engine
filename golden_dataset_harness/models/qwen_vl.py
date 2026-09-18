"""Qwen VL Model integration via OpenAI-compatible Vision API.

Works with any server exposing an OpenAI-compatible vision chat completion endpoint:
- Local vLLM (`vllm serve Qwen/Qwen2.5-VL-7B-Instruct`)
- Local Ollama (`ollama run qwen2.5-vl`)
- Remote APIs (OpenRouter, DashScope, OpenAI-compatible vision endpoints)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re

import httpx

from golden_dataset_harness.models.base import BaseVisionLanguageModel
from golden_dataset_harness.schemas.annotation import GroundingResult, QualityJudgment

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract and parse JSON from model output, handling markdown fences and commentary."""
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to extract valid JSON from response: %r", text[:100])
    return {}


class QwenVLModel(BaseVisionLanguageModel):
    """Real Vision Language Model implementation using OpenAI-compatible Vision API."""

    @property
    def model_id(self) -> str:
        return self.model_name or "Qwen2.5-VL"

    def _get_headers(self) -> dict[str, str]:
        """Resolve API authorization headers."""
        key = self.api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _prepare_messages(self, image: bytes, text: str) -> list[dict]:
        """Format the image and text into an OpenAI-compatible vision payload."""
        b64_image = base64.b64encode(image).decode("utf-8")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

    async def _call_api(self, messages: list[dict], temperature: float = 0.2, **kwargs) -> str:
        """Call the VLM API with exponential backoff retry logic."""
        headers = self._get_headers()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        url = f"{self.api_base_url.rstrip('/')}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content if content is not None else ""
            except Exception as e:
                logger.warning(
                    "VLM API call to %s failed (attempt %d/%d): %s",
                    url, attempt + 1, self.max_retries, e,
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        return ""

    async def generate_caption(self, image: bytes, prompt: str | None = None) -> str:
        """Generate a factual, concise description of the person in the image."""
        text = (
            prompt
            or "Describe the person in this image concisely, focusing on gender, clothing style, colors, and accessories."
        )
        messages = self._prepare_messages(image, text)
        result = await self._call_api(messages, temperature=0.7)
        return result.strip()

    async def extract_attributes(
        self, image: bytes, taxonomy: dict[str, list[str]]
    ) -> dict[str, str]:
        """Extract attributes constrained strictly to the allowed taxonomy values."""
        text = (
            "You are an expert CCTV and person search annotator. "
            "Analyze the image and extract the person's attributes strictly according to this JSON schema and allowed values:\n"
            f"{json.dumps(taxonomy, indent=2)}\n\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON object with matching keys.\n"
            "2. For each key, choose EXACTLY one value from the allowed list. If not visible or uncertain, choose 'unknown'.\n"
            "3. Do not include any explanation or extra text."
        )
        messages = self._prepare_messages(image, text)
        response_text = await self._call_api(messages, temperature=0.1)
        return _extract_json(response_text)

    async def verify_claim(self, image: bytes, claim: str) -> GroundingResult:
        """Verify whether a specific claim about the person is supported by the image."""
        text = (
            f"Carefully inspect the image to verify this claim: '{claim}'.\n"
            "Output a JSON object with this exact structure:\n"
            "{\n"
            '  "supported": true or false,\n'
            '  "confidence": number between 0.0 and 1.0,\n'
            '  "reason": "short explanation of visual evidence"\n'
            "}\n"
            "If the item or attribute is missing, obscured, or cannot be seen, supported MUST be false."
        )
        messages = self._prepare_messages(image, text)
        response_text = await self._call_api(messages, temperature=0.1)
        data = _extract_json(response_text)

        supported = bool(data.get("supported", False))
        confidence = float(data.get("confidence", 0.5))
        reason = str(data.get("reason") or data.get("explanation") or "")

        return GroundingResult(
            claim=claim,
            supported=supported,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason,
        )

    async def judge_quality(
        self, image: bytes, caption: str, attributes: dict
    ) -> QualityJudgment:
        """Judge annotation consistency, correctness, and hallucination."""
        text = (
            "You are an independent quality auditor for an AI surveillance dataset.\n"
            f"Given the image, evaluate this annotation:\n"
            f"- Caption: {caption}\n"
            f"- Attributes: {json.dumps(attributes)}\n\n"
            "Evaluate:\n"
            "1. Consistency between the caption and image.\n"
            "2. Accuracy of the extracted attributes.\n"
            "3. Absence of hallucinations (objects mentioned that are not visible).\n"
            "4. Completeness of salient features.\n\n"
            "Output ONLY a JSON object:\n"
            "{\n"
            '  "quality_score": number between 0.0 and 1.0,\n'
            '  "issues": ["list of identified issues, or empty if perfect"]\n'
            "}"
        )
        messages = self._prepare_messages(image, text)
        response_text = await self._call_api(messages, temperature=0.1)
        data = _extract_json(response_text)

        quality_score = float(data.get("quality_score", 0.7))
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]

        return QualityJudgment(
            quality_score=max(0.0, min(1.0, quality_score)),
            issues=[str(i) for i in issues],
        )
