"""Shared SigLIP attribute contract for extraction, validation and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml


class AttributeDefinition(TypedDict):
    type: Literal["single_label", "multi_label"]
    classes: list[str]
    label: str


Taxonomy = dict[str, AttributeDefinition]
AttributeValue = str | list[str] | None
DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "configs" / "attributes.yaml"


def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    with open(path, encoding="utf-8") as source:
        taxonomy = yaml.safe_load(source)
    if not isinstance(taxonomy, dict) or not taxonomy:
        raise ValueError("Attribute taxonomy must be a nonempty mapping")
    for name, definition in taxonomy.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Attribute {name} needs type and classes")
        classes = definition.get("classes")
        if definition.get("type") not in {"single_label", "multi_label"}:
            raise ValueError(f"Invalid attribute type for {name}")
        if (not isinstance(classes, list) or not classes
                or any(not isinstance(code, str) or not code for code in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError(f"Invalid classes for {name}")
    return taxonomy


TAXONOMY = load_taxonomy()


def validate_attributes(
    raw: dict[str, Any], taxonomy: Taxonomy = TAXONOMY, *, require_all: bool = False,
) -> dict[str, AttributeValue]:
    """Reject invalid labels instead of inventing unknown or negative labels."""
    extra = raw.keys() - taxonomy.keys()
    if extra:
        raise ValueError(f"Unexpected attributes: {sorted(extra)}")
    result: dict[str, AttributeValue] = {}
    for name, definition in taxonomy.items():
        if name not in raw:
            if require_all:
                raise ValueError(f"Missing attribute {name}")
            continue
        value = raw[name]
        if value is None:
            result[name] = None
            continue
        if definition["type"] == "multi_label":
            valid = (isinstance(value, list)
                     and all(isinstance(v, str) and v in definition["classes"] for v in value)
                     and len(value) == len(set(value)))
        else:
            valid = isinstance(value, str) and value in definition["classes"]
        if not valid:
            raise ValueError(f"Invalid value returned for attribute {name}: {value!r}")
        result[name] = value
    return result


def attributes_json_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    properties = {}
    for name, definition in taxonomy.items():
        choice = {"type": "string", "enum": definition["classes"]}
        value_schema = (
            {"type": "array", "items": choice, "uniqueItems": True}
            if definition["type"] == "multi_label" else choice
        )
        properties[name] = {"anyOf": [value_schema, {"type": "null"}]}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def attribute_cells(values: dict[str, AttributeValue]) -> dict[str, str]:
    """SigLIP table format: null -> blank; [] -> none; labels joined by |."""
    validate_attributes(values)
    return {
        name: "" if value is None else ("|".join(value) if value else "none")
        if isinstance(value, list) else value
        for name, value in values.items()
    }


def parse_attribute_cells(cells: dict[str, str]) -> dict[str, AttributeValue]:
    values: dict[str, AttributeValue] = {}
    for name, cell in cells.items():
        if name not in TAXONOMY:
            raise ValueError(f"Unexpected attribute {name}")
        cell = cell.strip()
        if not cell:
            values[name] = None
        elif TAXONOMY[name]["type"] == "multi_label":
            values[name] = [] if cell == "none" else [v.strip() for v in cell.split("|")]
        else:
            values[name] = cell
    return validate_attributes(values)
