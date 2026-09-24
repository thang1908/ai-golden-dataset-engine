#!/usr/bin/env python3
"""Refresh G1–G5 predictions in an existing caption-review CSV.

The existing CSV is the source of the golden captions and sample-to-image mapping.
SQLite is only a temporary on-disk join index, so the source dataset does not need
to fit in RAM. No pairs_*.tsv file is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from pathlib import Path


VARIANTS = (
    ("en", "short"),
    ("en", "medium"),
    ("en", "long"),
    ("vi", "short"),
    ("vi", "medium"),
    ("vi", "long"),
)
METHODS = ("g1", "g2", "g3", "g4", "g5")


def load_golden(connection: sqlite3.Connection, source_path: Path) -> None:
    """Load the immutable golden-caption and image mapping columns from CSV."""
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing caption review source: {source_path}. "
            "This file preserves golden captions after pairs_*.tsv was removed."
        )
    golden_columns = [f"golden_{language}_{length}" for language, length in VARIANTS]
    required = {"sample_id", "image_path", *golden_columns}
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Caption review source has missing required columns: {source_path}")
        statement = f"""
            INSERT INTO captions (sample_id, image_path, {', '.join(golden_columns)})
            VALUES ({', '.join('?' for _ in range(2 + len(golden_columns)))})
        """
        for line_number, row in enumerate(reader, start=2):
            sample_id, image_path = row.get("sample_id", ""), row.get("image_path", "")
            if not sample_id or not image_path:
                raise ValueError(f"Missing sample_id or image_path in {source_path}:{line_number}")
            try:
                connection.execute(
                    statement,
                    (sample_id, image_path, *(row.get(column, "") for column in golden_columns)),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate sample_id in {source_path}:{line_number}") from exc
    connection.commit()


def load_predictions(connection: sqlite3.Connection, outputs_root: Path) -> dict[str, int]:
    matched: dict[str, int] = {}
    for method in METHODS:
        path = outputs_root / method / "predictions.jsonl"
        matched[method] = 0
        if not path.is_file():
            continue
        statement = f"""
            UPDATE captions
            SET {method}_caption_en = ?, {method}_caption_vi = ?, {method}_status = ?
            WHERE sample_id = ?
        """
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                sample_id = str(row.get("sample_id", ""))
                if not sample_id:
                    continue
                cursor = connection.execute(
                    statement,
                    (
                        row.get("caption", ""),
                        row.get("caption_vi", ""),
                        row.get("status", ""),
                        sample_id,
                    ),
                )
                matched[method] += cursor.rowcount
        connection.commit()
    return matched


def write_csv(connection: sqlite3.Connection, output_path: Path) -> int:
    headers = [
        "sample_id",
        "image_path",
        *(f"golden_{language}_{length}" for language, length in VARIANTS),
        *(
            field
            for method in METHODS
            for field in (
                f"{method}_caption_en",
                f"{method}_caption_vi",
                f"{method}_status",
            )
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        cursor = connection.execute(f"SELECT {', '.join(headers)} FROM captions ORDER BY sample_id")
        count = 0
        for row in cursor:
            writer.writerow("" if value is None else value for value in row)
            count += 1
    return count


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Refresh G1–G5 captions using an existing caption-review CSV as golden source."
        )
    )
    parser.add_argument("--outputs-root", type=Path, default=root / "output")
    parser.add_argument(
        "--golden-source",
        type=Path,
        default=root / "output" / "review" / "captions_merged.csv",
        help="Existing CSV containing golden caption columns and image_path mapping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "review" / "captions_merged.csv",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="caption-merge-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "join.sqlite3")
        try:
            columns = ", ".join(
                ["sample_id TEXT PRIMARY KEY", "image_path TEXT"]
                + [f"golden_{language}_{length} TEXT" for language, length in VARIANTS]
                + [
                    f"{method}_{field} TEXT"
                    for method in METHODS
                    for field in ("caption_en", "caption_vi", "status")
                ]
            )
            connection.execute(f"CREATE TABLE captions ({columns})")
            load_golden(connection, args.golden_source)
            matched = load_predictions(connection, args.outputs_root)
            rows = write_csv(connection, args.output)
        finally:
            connection.close()

    summary = ", ".join(f"{method}={count}" for method, count in matched.items())
    print(f"Refreshed {rows} caption rows to {args.output} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
