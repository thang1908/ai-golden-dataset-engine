from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ResponseValidationError

ATTRIBUTE_FIELDS = (
    "age", "gender", "body_build", "upper_clothing_type", "upper_clothing_color",
    "lower_clothing_type", "lower_clothing_color", "clothing_style", "upper_pattern",
    "footwear_type", "bag_type", "bag_color", "headwear", "eyewear", "face_mask",
    "other_accessories", "hair_length", "hair_texture", "hairstyle", "hair_color",
    "carried_objects",
)
MAX_NOTE_LENGTH = 400


def caption_factuality_schema() -> dict[str, Any]:
    """Strict provider schema for image-grounded caption factuality only."""
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "is_correct": {"type": "boolean"},
            "needs_human_review": {"type": "boolean"},
            "note": {"type": "string", "maxLength": MAX_NOTE_LENGTH},
        },
        "required": ["is_correct", "needs_human_review", "note"],
    }


def caption_attribute_schema() -> dict[str, Any]:
    """Strict provider schema for caption claims matched against golden fields."""
    unmentioned_result = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "mentioned": {"type": "boolean", "enum": [False]},
            "is_correct": {"type": "null"},
            "note": {"type": "string", "maxLength": MAX_NOTE_LENGTH},
        },
        "required": ["mentioned", "is_correct", "note"],
    }
    mentioned_result = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "mentioned": {"type": "boolean", "enum": [True]},
            "is_correct": {"type": "boolean"},
            "note": {"type": "string", "maxLength": MAX_NOTE_LENGTH},
        },
        "required": ["mentioned", "is_correct", "note"],
    }
    field_result = {
        "anyOf": [unmentioned_result, mentioned_result],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {field: field_result for field in ATTRIBUTE_FIELDS},
        "required": list(ATTRIBUTE_FIELDS),
    }


def validate_prediction_attributes(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(ATTRIBUTE_FIELDS):
        raise ResponseValidationError(
            "Prediction attributes must contain exactly the 21 taxonomy fields"
        )
    return value


def _is_boolean(value: object) -> bool:
    return isinstance(value, bool)


def _is_note(value: object) -> bool:
    return isinstance(value, str) and len(value) <= MAX_NOTE_LENGTH


def validate_caption_factuality(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "is_correct", "needs_human_review", "note",
    }:
        raise ResponseValidationError("Caption factuality response has an invalid shape")
    if (
        not _is_boolean(value["is_correct"])
        or not _is_boolean(value["needs_human_review"])
        or not _is_note(value["note"])
    ):
        raise ResponseValidationError("Caption factuality response has invalid values")
    return dict(value)


def validate_caption_attribute_evaluation(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(ATTRIBUTE_FIELDS):
        raise ResponseValidationError("Caption attribute response must contain all 21 fields")
    validated: dict[str, dict[str, Any]] = {}
    for field in ATTRIBUTE_FIELDS:
        item = value[field]
        if not isinstance(item, Mapping) or set(item) != {"mentioned", "is_correct", "note"}:
            raise ResponseValidationError(f"Caption attribute response has invalid field {field}")
        mentioned = item["mentioned"]
        correctness = item["is_correct"]
        if (
            not _is_boolean(mentioned)
            or not _is_note(item["note"])
        ):
            raise ResponseValidationError(f"Caption attribute response has invalid values for {field}")
        if mentioned and not _is_boolean(correctness):
            raise ResponseValidationError(
                f"Caption attribute {field} must be true or false when it is mentioned"
            )
        if not mentioned and correctness is not None:
            raise ResponseValidationError(
                f"Caption attribute {field} must be null when it is not mentioned"
            )
        validated[field] = dict(item)
    return validated
