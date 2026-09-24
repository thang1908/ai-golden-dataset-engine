from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ATTRIBUTE_FIELDS, validate_prediction_attributes
from .errors import DatasetError


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    sample_id: str
    method: str
    image_path: Path
    relative_image_path: str
    image_id: str
    golden_attributes: dict[str, Any]
    prediction_caption: str
    prediction_caption_vi: str | None
    prediction_attributes: dict[str, Any]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"Missing dataset manifest: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _golden_attributes(test_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_tsv(test_dir / "attributes.tsv")
    required = {"person_id", *ATTRIBUTE_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise DatasetError("attributes.tsv does not contain the required 21 attributes")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("person_id", "")
        if not sample_id or sample_id in result:
            raise DatasetError("attributes.tsv has missing or duplicate person_id")
        result[sample_id] = {field: row[field] for field in ATTRIBUTE_FIELDS}
    return result


def _image_mappings(
    test_dir: Path,
    outputs_root: Path,
    attribute_ids: set[str],
) -> dict[str, dict[str, str]]:
    manifest = outputs_root / "review" / "captions_merged.csv"
    if not manifest.is_file():
        raise DatasetError(f"Missing image mapping: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id, relative = row.get("sample_id", ""), row.get("image_path", "")
        image_path = test_dir / relative
        if (
            not sample_id
            or sample_id in result
            or sample_id not in attribute_ids
            or not relative
            or not image_path.is_file()
        ):
            raise DatasetError("captions_merged.csv has an invalid or ambiguous image mapping")
        result[sample_id] = row
    if not result:
        raise DatasetError("captions_merged.csv contains no image mappings")
    return result


def _predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON in {path}:{number}") from exc
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise DatasetError(f"Duplicate or missing sample_id in {path}:{number}")
            seen.add(sample_id)
            result.append(row)
    return result


def load_tasks(
    test_dir: Path,
    outputs_root: Path,
    methods: tuple[str, ...],
) -> list[EvaluationTask]:
    golden = _golden_attributes(test_dir)
    images = _image_mappings(test_dir, outputs_root, set(golden))
    tasks: list[EvaluationTask] = []
    for method in methods:
        for prediction in _predictions(outputs_root / method / "predictions.jsonl"):
            if prediction.get("status") != "success":
                continue
            sample_id = str(prediction.get("sample_id", ""))
            if sample_id not in images:
                raise DatasetError(f"{method} prediction has no mapped query image: {sample_id}")
            caption = prediction.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise DatasetError(f"{method} successful prediction {sample_id} has no caption")
            attributes = validate_prediction_attributes(prediction.get("attributes"))
            image = images[sample_id]
            relative = image["image_path"]
            tasks.append(
                EvaluationTask(
                    sample_id=sample_id,
                    method=method,
                    image_path=test_dir / relative,
                    relative_image_path=relative,
                    image_id=relative,
                    golden_attributes=golden[sample_id],
                    prediction_caption=caption.strip(),
                    prediction_caption_vi=(
                        prediction["caption_vi"].strip()
                        if isinstance(prediction.get("caption_vi"), str)
                        and prediction["caption_vi"].strip()
                        else None
                    ),
                    prediction_attributes=attributes,
                )
            )
    return tasks
