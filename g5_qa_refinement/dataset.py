"""Read the existing sample's canonical query list and write matching JSONL rows."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, G5Error
from .taxonomy import TAXONOMY


@dataclass(frozen=True, slots=True)
class TestSample:
    sample_id: str
    image_path: Path
    relative_image_path: str


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ConfigurationError(f"Missing dataset file: {path}")
    return list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))


def load_query_samples(test_dir: Path) -> list[TestSample]:
    pairs = _read_tsv(test_dir / "pairs_en_medium.tsv")
    attributes = _read_tsv(test_dir / "attributes.tsv")
    if not attributes or not set(TAXONOMY).issubset(attributes[0]):
        raise ConfigurationError("attributes.tsv does not match the required 21-attribute schema")
    attribute_ids = {row.get("person_id", "") for row in attributes}
    result: list[TestSample] = []
    seen: set[str] = set()
    for row in pairs:
        if row.get("is_query") != "1":
            continue
        sample_id, relative = row.get("person_id", ""), row.get("filepath", "")
        path = test_dir / relative
        if (
            not sample_id
            or sample_id in seen
            or sample_id not in attribute_ids
            or not path.is_file()
        ):
            raise ConfigurationError("pairs_en_medium.tsv has an invalid query row")
        seen.add(sample_id)
        result.append(TestSample(sample_id, path, relative))
    if not result:
        raise ConfigurationError("pairs_en_medium.tsv has no query images")
    return result


def prediction_row(
    sample: TestSample,
    annotation: dict[str, Any] | None,
    *,
    model: str,
    latency_ms: float,
    run_id: str | None = None,
    telemetry: dict[str, Any] | None = None,
    error: str | None = None,
    error_code: str = "generation_failed",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "method": "g5_qa_refinement",
        "model_id": model,
        "prompt_version": "qa_refinement_v3",
        "latency_ms": round(latency_ms, 2),
        "status": "success" if error is None else "error",
    }
    if run_id is not None:
        row["run_id"] = run_id
    if telemetry is not None:
        row.update(telemetry)
    if annotation is not None:
        row["caption"] = annotation["caption"]
        row["caption_vi"] = annotation["caption_vi"]
        row["attributes"] = annotation["attributes"]
        row["workflow_status"] = annotation.get("workflow_status", "accepted")
        row["workflow_notes"] = annotation.get("workflow_notes", [])
    else:
        row["error"] = {"code": error_code, "message": error or "Generation failed"}
    return row


def generate_dataset(samples: list[TestSample], annotate, *, model: str, run_id: str, metrics_for_sample=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        try:
            annotation = annotate(sample.image_path.read_bytes(), sample.sample_id, run_id)
            rows.append(
                prediction_row(
                    sample,
                    annotation,
                    model=model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    run_id=run_id,
                    telemetry=metrics_for_sample(sample.sample_id) if metrics_for_sample else None,
                )
            )
        except G5Error as exc:
            rows.append(
                prediction_row(
                    sample,
                    None,
                    model=model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    run_id=run_id,
                    telemetry=metrics_for_sample(sample.sample_id) if metrics_for_sample else None,
                    error=str(exc),
                    error_code=str(exc).split(":", 1)[0].replace(" ", "_").lower(),
                )
            )
        except Exception:
            rows.append(
                prediction_row(
                    sample,
                    None,
                    model=model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    run_id=run_id,
                    telemetry=metrics_for_sample(sample.sample_id) if metrics_for_sample else None,
                    error="An unexpected local generation error occurred",
                )
            )
    return rows
