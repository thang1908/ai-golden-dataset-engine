from __future__ import annotations

from typing import Any

from .client import VllmClient
from .agents.nodes import caption_agent, consensus_agent, observer_agent, vietnamese_translation_agent
from .errors import ResponseValidationError
from .prompts import caption_prompt, consensus_prompt, observer_prompt
from .taxonomy import attribute_json_schema, validate_attributes
from .translation import translate_caption

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "minLength": 1, "maxLength": 500},
        "attributes": attribute_json_schema(),
    },
    "required": ["caption", "attributes"],
    "additionalProperties": False,
}
CONSENSUS_SCHEMA = {
    "type": "object",
    "properties": {"attributes": attribute_json_schema()},
    "required": ["attributes"],
    "additionalProperties": False,
}
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {"caption": {"type": "string", "minLength": 1, "maxLength": 500}},
    "required": ["caption"],
    "additionalProperties": False,
}


def _annotation(value: dict[str, Any]) -> dict[str, Any]:
    try:
        caption = value["caption"]
        if set(value) != {"caption", "attributes"} or not isinstance(caption, str):
            raise ValueError
        caption = caption.strip()
        if not caption or len(caption) > 500:
            raise ValueError
        return {"caption": caption, "attributes": validate_attributes(value["attributes"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError("Model returned an invalid observer annotation") from exc


def run(image: bytes, client: VllmClient) -> dict[str, Any]:
    candidates = [
        _annotation(
            observer_agent(index, image, client, ANNOTATION_SCHEMA)
        )
        for index in range(4)
    ]
    consensus = consensus_agent(image, candidates, client, CONSENSUS_SCHEMA)
    try:
        if set(consensus) != {"attributes"}:
            raise ValueError
        attributes = validate_attributes(consensus["attributes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError("Model returned invalid consensus attributes") from exc
    caption_response = caption_agent(attributes, client, CAPTION_SCHEMA)
    caption = caption_response.get("caption")
    if not isinstance(caption, str) or not (caption := caption.strip()) or len(caption) > 500:
        raise ResponseValidationError("Model returned an invalid caption")
    return {
        "caption": caption,
        "caption_vi": vietnamese_translation_agent(caption, client),
        "attributes": attributes,
    }
