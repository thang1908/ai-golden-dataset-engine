from __future__ import annotations

from typing import Any

from .client import VllmClient
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
            client.complete(
                prompt=observer_prompt(index),
                schema_name=f"observer_{index + 1}",
                schema=ANNOTATION_SCHEMA,
                image=image,
            )
        )
        for index in range(4)
    ]
    consensus = client.complete(
        prompt=consensus_prompt(candidates),
        schema_name="attribute_consensus",
        schema=CONSENSUS_SCHEMA,
        image=image,
    )
    try:
        if set(consensus) != {"attributes"}:
            raise ValueError
        attributes = validate_attributes(consensus["attributes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError("Model returned invalid consensus attributes") from exc
    caption_response = client.complete(
        prompt=caption_prompt(attributes), schema_name="caption", schema=CAPTION_SCHEMA
    )
    caption = caption_response.get("caption")
    if not isinstance(caption, str) or not (caption := caption.strip()) or len(caption) > 500:
        raise ResponseValidationError("Model returned an invalid caption")
    return {
        "caption": caption,
        "caption_vi": translate_caption(client, caption),
        "attributes": attributes,
    }
