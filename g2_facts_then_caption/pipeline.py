from __future__ import annotations

from typing import Any

from .client import VllmClient
from .errors import ResponseValidationError
from .prompts import caption_prompt, structured_facts_prompt, visual_analysis_prompt
from .taxonomy import attribute_json_schema, validate_attributes
from .translation import translate_caption

_STRING_LIST = {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 300}, "maxItems": 30}
VISUAL_ANALYSIS_SCHEMA = {"type": "object", "properties": {"observations": _STRING_LIST, "uncertainties": _STRING_LIST}, "required": ["observations", "uncertainties"], "additionalProperties": False}
STRUCTURED_FACTS_SCHEMA = {"type": "object", "properties": {"facts": _STRING_LIST, "attributes": attribute_json_schema()}, "required": ["facts", "attributes"], "additionalProperties": False}
CAPTION_SCHEMA = {"type": "object", "properties": {"caption": {"type": "string", "minLength": 1, "maxLength": 500}}, "required": ["caption"], "additionalProperties": False}


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ResponseValidationError(f"Model returned invalid {field}")
    return [item.strip() for item in value]


def run(image: bytes, client: VllmClient) -> dict[str, Any]:
    analysis = client.complete(prompt=visual_analysis_prompt(), schema_name="visual_analysis", schema=VISUAL_ANALYSIS_SCHEMA, image=image)
    observations = _strings(analysis.get("observations"), "observations")
    uncertainties = _strings(analysis.get("uncertainties"), "uncertainties")
    normalized_analysis = {"observations": observations, "uncertainties": uncertainties}

    facts_response = client.complete(prompt=structured_facts_prompt(normalized_analysis), schema_name="structured_facts", schema=STRUCTURED_FACTS_SCHEMA, image=image)
    facts = _strings(facts_response.get("facts"), "facts")
    try:
        attributes = validate_attributes(facts_response.get("attributes"))
    except ValueError as exc:
        raise ResponseValidationError("Model returned invalid structured attributes") from exc

    caption_response = client.complete(prompt=caption_prompt(facts), schema_name="caption", schema=CAPTION_SCHEMA)
    caption = caption_response.get("caption")
    if not isinstance(caption, str) or not (caption := caption.strip()) or len(caption) > 500:
        raise ResponseValidationError("Model returned an invalid caption")
    return {
        "caption": caption,
        "caption_vi": translate_caption(client, caption),
        "attributes": attributes,
    }
