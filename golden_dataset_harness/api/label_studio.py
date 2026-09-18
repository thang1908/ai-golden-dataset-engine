"""Label Studio integration — export and import helpers.

Formats annotations for Label Studio's JSON import format, including
pre-annotations and judge feedback for efficient human review.
"""

from __future__ import annotations

from typing import Any

from golden_dataset_harness.schemas.annotation import AnnotationRecord


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
                    {
                        "id": f"attrs_{record.image_id}",
                        "type": "choices",
                        "value": {
                            "choices": [
                                f"{k}: {v}"
                                for k, v in record.attributes.model_dump().items()
                                if v != "unknown"
                            ],
                        },
                        "from_name": "attributes",
                        "to_name": "image",
                    },
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
        corrected_attrs: dict[str, str] = {}

        for item in result:
            if item.get("from_name") == "caption":
                texts = item.get("value", {}).get("text", [])
                if texts:
                    corrected_caption = texts[0]
            elif item.get("from_name") == "attributes":
                choices = item.get("value", {}).get("choices", [])
                for choice in choices:
                    if ": " in choice:
                        k, v = choice.split(": ", 1)
                        corrected_attrs[k] = v

        reviews.append({
            "image_id": image_id,
            "status": "human_approved",
            "corrected_caption": corrected_caption,
            "corrected_attributes": corrected_attrs,
            "reviewer_notes": latest.get("lead_time", ""),
        })

    return reviews
