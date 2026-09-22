"""Sequential image execution and reproducible local output for both baselines."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from golden_dataset_harness.baselines.contracts import (
    InferenceResult,
    Method,
    PredictionError,
    PredictionRecord,
)
from golden_dataset_harness.schemas.taxonomy import TAXONOMY

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BaselineConfigError(ValueError):
    """Safe, user-facing setup error (never contains provider credentials)."""


class Predictor(Protocol):
    method: Method
    model_id: str
    model_calls: int
    http_attempts: int
    metadata: dict[str, Any]

    async def predict(self, image: bytes) -> InferenceResult: ...


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One person image/crop")
    source.add_argument("--input-dir", type=Path, help="Image directory (not recursive)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing run files")


def discover_images(image: Path | None, input_dir: Path | None) -> list[Path]:
    if (image is None) == (input_dir is None):
        raise BaselineConfigError("Specify exactly one of --image or --input-dir")
    if image is not None:
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise BaselineConfigError("--image must be an existing supported image file")
        return [image]
    if not input_dir.is_dir():
        raise BaselineConfigError("--input-dir must be an existing directory")
    paths = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise BaselineConfigError("No supported images in --input-dir")
    return paths


def check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise BaselineConfigError("--output-dir is not a directory")
    if path.exists() and not overwrite:
        raise BaselineConfigError("Output directory already exists; use a new path or --overwrite")


def _safe_error(exc: Exception) -> PredictionError:
    # Never serialize exception text: HTTP and validation errors can embed inputs/secrets.
    if isinstance(exc, (UnidentifiedImageError, Image.DecompressionBombError)):
        return PredictionError(code="invalid_image", message="Input is not a supported image")
    if isinstance(exc, OSError):
        return PredictionError(code="image_io_error", message="Unable to read or decode image")
    if isinstance(exc, (ValidationError, ValueError)):
        return PredictionError(code="invalid_prediction", message="Image or prediction is invalid")
    return PredictionError(code="inference_error", message="Model inference failed")


def _write_summary(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".summary-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def run_images(
    predictor: Predictor,
    paths: Sequence[Path],
    output_dir: Path,
    *,
    overwrite: bool = False,
    model_load_ms: float = 0,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Persist every input, including failures; flush completed rows on interruption."""
    if not paths:
        raise BaselineConfigError("No input images")
    check_output(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    summary_path = output_dir / "summary.json"
    # A previous completed summary must not describe an interrupted replacement run.
    if overwrite:
        summary_path.unlink(missing_ok=True)
    run_id = str(uuid.uuid4())
    started_at = started_at or datetime.now(UTC).isoformat()
    run_start = time.perf_counter()
    latencies: list[float] = []
    errors = 0
    total_calls = 0
    total_attempts = 0
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for path in paths:
            start = time.perf_counter()
            before_calls, before_attempts = predictor.model_calls, predictor.http_attempts
            image_hash = None
            result = None
            error = None
            try:
                image = path.read_bytes()
                image_hash = hashlib.sha256(image).hexdigest()
                with Image.open(io.BytesIO(image)) as source:
                    source.verify()
                result = await predictor.predict(image)
                # Revalidate even a custom predictor before writing a successful row.
                result = InferenceResult.model_validate(result.model_dump())
            except Exception as exc:
                error = _safe_error(exc)
                errors += 1
            elapsed = (time.perf_counter() - start) * 1000
            if error is None:
                latencies.append(elapsed)
            calls = predictor.model_calls - before_calls
            attempts = predictor.http_attempts - before_attempts
            total_calls += calls
            total_attempts += attempts
            record = PredictionRecord(
                run_id=run_id, image_id=path.name, image_path=str(path),
                image_sha256=image_hash, method=predictor.method, model_id=predictor.model_id,
                status="error" if error else "success",
                attributes=result.attributes if result is not None and error is None else None,
                scores=result.scores if result is not None and error is None else None,
                elapsed_ms=elapsed, model_calls=calls, http_attempts=attempts, error=error,
            )
            stream.write(record.model_dump_json() + "\n")
            stream.flush()
    summary = {
        "schema_version": "baseline-v1", "attribute_schema_version": "siglip-v1",
        "run_id": run_id, "method": predictor.method, "model_id": predictor.model_id,
        "started_at": started_at, "completed_at": datetime.now(UTC).isoformat(),
        "num_inputs": len(paths), "success_count": len(latencies), "error_count": errors,
        "model_load_ms": model_load_ms,
        "total_elapsed_ms": model_load_ms + (time.perf_counter() - run_start) * 1000,
        "latency_ms_success": {
            "mean": statistics.mean(latencies), "median": statistics.median(latencies),
            "min": min(latencies), "max": max(latencies),
        } if latencies else None,
        "model_calls": total_calls, "http_attempts": total_attempts,
        "taxonomy_sha256": json_hash(TAXONOMY), "config": predictor.metadata,
        "input_files": [str(path) for path in paths],
    }
    _write_summary(summary_path, summary)
    return summary


def print_summary(summary: dict[str, Any], output_dir: Path) -> int:
    print(f"{summary['method']}: {summary['success_count']}/{summary['num_inputs']} succeeded, "
          f"{summary['error_count']} failed. Output: {output_dir}")
    return 1 if summary["error_count"] else 0
