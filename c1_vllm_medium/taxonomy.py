"""The self-contained 21-attribute contract used by C1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("attribute_schema.json")


def load_taxonomy() -> dict[str, dict[str, Any]]:
    """Load the vendored SigLIP-compatible codes without importing another project."""
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    attributes = payload.get("attributes")
    if not isinstance(attributes, list) or len(attributes) != 21:
        raise ValueError("C1 attribute schema must contain exactly 21 attributes")
    taxonomy: dict[str, dict[str, Any]] = {}
    for entry in attributes:
        name, kind, classes = entry.get("name"), entry.get("type"), entry.get("classes")
        if (not isinstance(name, str) or kind not in {"single_label", "multi_label"}
                or not isinstance(classes, list) or not classes
                or not all(isinstance(code, str) for code in classes)):
            raise ValueError("C1 attribute schema is invalid")
        taxonomy[name] = {"type": kind, "classes": classes}
    if len(taxonomy) != 21:
        raise ValueError("C1 attribute names must be unique")
    return taxonomy


TAXONOMY = load_taxonomy()


def json_schema() -> dict[str, Any]:
    """OpenAI-compatible JSON Schema that requires all 21 attributes."""
    properties: dict[str, Any] = {}
    for name, definition in TAXONOMY.items():
        choice = {"type": "string", "enum": definition["classes"]}
        value = (
            {"type": "array", "items": choice, "uniqueItems": True}
            if definition["type"] == "multi_label" else choice
        )
        properties[name] = {"anyOf": [value, {"type": "null"}]}
    return {
        "type": "object", "properties": properties, "required": list(TAXONOMY),
        "additionalProperties": False,
    }


def validate_attributes(value: Any) -> dict[str, str | list[str] | None]:
    """Validate the model response after JSON Schema enforcement."""
    if not isinstance(value, dict) or set(value) != set(TAXONOMY):
        raise ValueError("Response must contain exactly the 21 configured attributes")
    result: dict[str, str | list[str] | None] = {}
    for name, definition in TAXONOMY.items():
        label = value[name]
        if label is None:
            result[name] = None
        elif definition["type"] == "single_label":
            if not isinstance(label, str) or label not in definition["classes"]:
                raise ValueError(f"Invalid value for {name}")
            result[name] = label
        else:
            if (not isinstance(label, list) or len(label) != len(set(label))
                    or not all(isinstance(code, str) and code in definition["classes"]
                               for code in label)):
                raise ValueError(f"Invalid values for {name}")
            result[name] = label
    return result
