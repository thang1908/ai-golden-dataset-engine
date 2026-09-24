#!/usr/bin/env python3
"""Build the local, read-only person-key-to-image index for the dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTRIBUTES = PROJECT_ROOT / "sample" / "test" / "attributes.tsv"
DEFAULT_CAPTIONS = PROJECT_ROOT / "output" / "review" / "captions_merged.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "dashboard" / "index.json"
TEST_ROOT = PROJECT_ROOT / "sample" / "test"


class IndexBuildError(Exception):
    """An authoritative source cannot produce a safe dashboard index."""


def read_rows(path: Path, delimiter: str) -> Iterable[dict[str, str]]:
    if not path.is_file():
        raise IndexBuildError(f"Missing source file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise IndexBuildError(f"Source has no header row: {path}")
        yield from reader


def require_columns(path: Path, rows: list[dict[str, str]], columns: set[str]) -> None:
    if not rows:
        raise IndexBuildError(f"Source has no data rows: {path}")
    actual = set(rows[0])
    missing = sorted(columns - actual)
    if missing:
        raise IndexBuildError(f"Source {path} is missing columns: {', '.join(missing)}")


def path_under_test_root(raw_path: str) -> tuple[Path, str]:
    """Return a safe test-relative file and its project-relative browser path."""
    if not raw_path:
        raise IndexBuildError("Mapped image_path is empty")

    candidate = (TEST_ROOT / raw_path).resolve()
    try:
        relative_to_test = candidate.relative_to(TEST_ROOT.resolve())
    except ValueError as exc:
        raise IndexBuildError(f"Mapped image_path escapes sample/test: {raw_path}") from exc
    if not candidate.is_file():
        raise IndexBuildError(f"Mapped image file does not exist: {candidate}")
    return candidate, (Path("sample") / "test" / relative_to_test).as_posix()


def build_index(attributes_path: Path, captions_path: Path) -> dict[str, object]:
    attribute_rows = list(read_rows(attributes_path, "\t"))
    caption_rows = list(read_rows(captions_path, ","))
    require_columns(attributes_path, attribute_rows, {"person_id", "person_key"})
    require_columns(captions_path, caption_rows, {"sample_id", "image_path"})

    person_to_sample: dict[str, str] = {}
    for row in attribute_rows:
        person_key = (row.get("person_key") or "").strip()
        sample_id = (row.get("person_id") or "").strip()
        if not person_key or not sample_id:
            raise IndexBuildError("attributes.tsv has a row without person_key or person_id")
        if person_key in person_to_sample:
            raise IndexBuildError(f"Duplicate person_key in attributes.tsv: {person_key}")
        person_to_sample[person_key] = sample_id

    sample_to_image: dict[str, str] = {}
    for row in caption_rows:
        sample_id = (row.get("sample_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not sample_id:
            raise IndexBuildError("captions_merged.csv has a row without sample_id")
        if sample_id in sample_to_image:
            raise IndexBuildError(f"Duplicate sample_id in captions_merged.csv: {sample_id}")
        sample_to_image[sample_id] = image_path

    items: dict[str, dict[str, str]] = {}
    for person_key, sample_id in person_to_sample.items():
        if sample_id not in sample_to_image:
            raise IndexBuildError(
                f"No image mapping for person_key {person_key} (sample_id {sample_id})"
            )
        _, project_relative_path = path_under_test_root(sample_to_image[sample_id])
        items[person_key] = {
            "person_key": person_key,
            "sample_id": sample_id,
            "image_path": project_relative_path,
        }

    return {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(items),
        "items": items,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the validated person-key-to-image index for the local dashboard."
    )
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = build_index(args.attributes.resolve(), args.captions.resolve())
        atomic_write_json(args.output.resolve(), payload)
    except IndexBuildError as exc:
        print(f"Dashboard index error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Dashboard index I/O error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {payload['item_count']} image mappings to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
