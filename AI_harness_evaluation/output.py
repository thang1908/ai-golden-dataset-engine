from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def completed_keys(
    path: Path,
    *,
    schema_version: str | None = None,
    caption_attribute_prompt_version: str | None = None,
) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    result: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, method = row.get("sample_id"), row.get("method")
            if (
                row.get("status") == "success"
                and (schema_version is None or row.get("evaluation_schema_version") == schema_version)
                and (
                    caption_attribute_prompt_version is None
                    or (row.get("prompt_versions") or {}).get("caption_attribute")
                    == caption_attribute_prompt_version
                )
                and isinstance(sample_id, str)
                and isinstance(method, str)
            ):
                result.add((sample_id, method))
    return result


def persist_row(stream, row: dict[str, Any]) -> None:
    stream.write(json_line(row))
    stream.flush()
    os.fsync(stream.fileno())
