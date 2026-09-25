from __future__ import annotations

from typing import Any

from .client import VllmClient
from .errors import ResponseValidationError
from .telemetry import CallContext, CaseMetrics

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"caption_vi": {"type": "string", "minLength": 1, "maxLength": 500}},
    "required": ["caption_vi"],
    "additionalProperties": False,
}


def translate_caption(client: VllmClient, caption: str, metrics: CaseMetrics | None = None, context: CallContext | None = None) -> str:
    prompt = (
        "Translate this factual person-image caption into natural Vietnamese. Preserve only "
        "the stated facts; do not add identity, relationships, location, intent, or details. "
        "Return only the requested JSON.\nEnglish caption:\n"
        + caption
    )
    value: dict[str, Any] = client.complete(
        prompt=prompt, schema_name="caption_vietnamese", schema=TRANSLATION_SCHEMA, metrics=metrics, context=context
    )
    caption_vi = value.get("caption_vi")
    if set(value) != {"caption_vi"} or not isinstance(caption_vi, str):
        raise ResponseValidationError("Model returned an invalid Vietnamese caption")
    caption_vi = caption_vi.strip()
    if not caption_vi or len(caption_vi) > 500:
        raise ResponseValidationError("Model returned an invalid Vietnamese caption")
    return caption_vi
