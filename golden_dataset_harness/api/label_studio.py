"""Label Studio integration — export and import helpers.

Formats annotations for Label Studio's JSON import format, including
pre-annotations and judge feedback for efficient human review.
"""

from __future__ import annotations

from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from golden_dataset_harness.schemas.annotation import AnnotationRecord
from golden_dataset_harness.schemas.taxonomy import TAXONOMY, parse_attribute_cells


def format_for_label_studio(
    records: list[AnnotationRecord],
    image_base_url: str,
) -> list[dict[str, Any]]:
    """Convert AnnotationRecords to Label Studio JSON import format.

    Each task includes:
    - The image URL
    - Pre-annotations (caption + attributes) for reviewer reference
    - Pipeline metadata (scores, issues) as additional context

    Args:
        records: List of AnnotationRecords to export.
        image_base_url: Base URL where images are accessible (e.g., MinIO presigned).

    Returns:
        List of Label Studio task dicts ready for JSON export.
    """
    tasks: list[dict[str, Any]] = []

    for record in records:
        # Build image URL
        image_url = f"{image_base_url.rstrip('/')}/{record.image_id}"
        if not any(image_url.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            image_url += ".jpg"

        # Pre-annotation: the pipeline's generated caption and attributes
        predictions = [
            {
                "model_version": record.model_version,
                "result": [
                    {
                        "id": f"caption_{record.image_id}",
                        "type": "textarea",
                        "value": {
                            "text": [record.caption],
                        },
                        "from_name": "caption",
                        "to_name": "image",
                    },
                    *[
                        {
                            "id": f"{key}_{record.image_id}",
                            "type": "choices",
                            "value": {"choices": (value or ["none"])
                                      if isinstance(value, list) else [value]},
                            "from_name": key,
                            "to_name": "image",
                        }
                        for key, value in record.attributes.model_dump().items()
                        if value is not None
                    ],
                ],
                "score": record.confidence,
            }
        ]

        task = {
            "data": {
                "image": image_url,
                "image_id": record.image_id,
                # Pipeline metadata for reviewer context
                "pipeline_caption": record.caption,
                "pipeline_attributes": record.attributes.model_dump(),
                "confidence_score": record.confidence,
                "judge_score": record.judge_score,
                "consensus_score": record.consensus_score,
                "grounding_score": record.grounding_score,
                "issues": record.issues,
            },
            "predictions": predictions,
        }

        tasks.append(task)

    return tasks


def parse_label_studio_export(
    export_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse Label Studio annotation export back into review decisions.

    Args:
        export_data: List of Label Studio completed task dicts.

    Returns:
        List of dicts with ``image_id``, ``status``, ``corrected_caption``,
        ``corrected_attributes``, and ``reviewer_notes``.
    """
    reviews: list[dict[str, Any]] = []

    for task in export_data:
        image_id = task.get("data", {}).get("image_id", "")
        annotations = task.get("annotations", [])

        if not annotations:
            continue

        # Take the latest annotation
        latest = annotations[-1]
        result = latest.get("result", [])

        corrected_caption = ""
        corrected_attrs: dict[str, str] = dict.fromkeys(TAXONOMY, "")

        for item in result:
            if item.get("from_name") == "caption":
                texts = item.get("value", {}).get("text", [])
                if texts:
                    corrected_caption = texts[0]
            elif item.get("from_name") in TAXONOMY:
                name = item["from_name"]
                choices = item.get("value", {}).get("choices", [])
                if TAXONOMY[name]["type"] == "single_label" and len(choices) > 1:
                    raise ValueError(f"Multiple choices for single-label attribute {name}")
                corrected_attrs[name] = "|".join(choices)

        reviews.append({
            "image_id": image_id,
            "status": "human_approved",
            "corrected_caption": corrected_caption,
            "corrected_attributes": parse_attribute_cells(corrected_attrs),
            "reviewer_notes": latest.get("lead_time", ""),
        })

    return reviews


def label_studio_config() -> str:
    """Generate controls matching the per-attribute predictions, including empty sets."""
    root = Element("View")
    SubElement(root, "Image", name="image", value="$image")
    SubElement(root, "TextArea", name="caption", toName="image", editable="true")
    for name, definition in TAXONOMY.items():
        SubElement(root, "Header", value=definition["label"])
        multiple = definition["type"] == "multi_label"
        control = SubElement(root, "Choices", name=name, toName="image",
                             choice="multiple" if multiple else "single")
        for code in definition["classes"]:
            SubElement(control, "Choice", value=code)
        if multiple:
            # Explicit negative label; a missing selection means unannotated.
            SubElement(control, "Choice", value="none")
    return tostring(root, encoding="unicode")
