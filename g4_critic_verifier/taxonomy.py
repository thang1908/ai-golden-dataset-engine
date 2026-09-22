from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("attribute_schema.json")


def _load() -> dict[str, dict[str, Any]]:
    entries = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")).get("attributes")
    if not isinstance(entries, list) or len(entries) != 21:
        raise ValueError("G4 attribute schema must contain exactly 21 attributes")
    result = {
        entry["name"]: {"type": entry["type"], "classes": entry["classes"]} for entry in entries
    }
    if len(result) != 21:
        raise ValueError("G4 attribute schema has duplicate names")
    return result


TAXONOMY = _load()


def attribute_json_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, definition in TAXONOMY.items():
        item = {"type": "string", "enum": definition["classes"]}
        value = (
            {"type": "array", "items": item, "uniqueItems": True}
            if definition["type"] == "multi_label"
            else item
        )
        properties[name] = {"anyOf": [value, {"type": "null"}]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(TAXONOMY),
        "additionalProperties": False,
    }


def validate_attributes(value: Any) -> dict[str, str | list[str] | None]:
    if not isinstance(value, dict) or set(value) != set(TAXONOMY):
        raise ValueError("Response must contain exactly the 21 configured attributes")
    output: dict[str, str | list[str] | None] = {}
    for name, definition in TAXONOMY.items():
        label = value[name]
        if label is None:
            output[name] = None
        elif (
            definition["type"] == "single_label"
            and isinstance(label, str)
            and label in definition["classes"]
        ):
            output[name] = label
        elif (
            definition["type"] == "multi_label"
            and isinstance(label, list)
            and len(label) == len(set(label))
            and all(isinstance(v, str) and v in definition["classes"] for v in label)
        ):
            output[name] = label
        else:
            raise ValueError(f"Invalid value for {name}")
    return output



