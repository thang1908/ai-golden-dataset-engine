"""Aggregate privacy-safe VLLM request telemetry into a benchmark report.

The generator writes one event for every HTTP attempt.  This tool deliberately
does not estimate tokens: totals are reported only when the VLLM response
included a standard ``usage`` object.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                result.append(row)
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def _summary(events: list[dict[str, Any]], predictions: list[dict[str, Any]], *, method: str, stage: str | None) -> dict[str, Any]:
    scoped_events = [item for item in events if item.get("method") == method and (stage is None or item.get("stage") == stage)]
    scoped_predictions = [item for item in predictions if item.get("method") in {method, {"g3": "g3_consensus_annotation", "g4": "g4_critic_verifier", "g5": "g5_qa_refinement"}.get(method)}]
    statuses = Counter(str(item.get("status")) for item in scoped_predictions)
    logical_calls = {(item.get("sample_id"), item.get("logical_call_id")) for item in scoped_events}
    usage_events = [item for item in scoped_events if item.get("usage_available")]
    token_keys = ("input_tokens", "output_tokens", "total_tokens")
    totals = {key: sum(int(item[key]) for item in usage_events if isinstance(item.get(key), (int, float))) for key in token_keys}
    latencies = [float(item["latency_ms"]) for item in scoped_events if isinstance(item.get("latency_ms"), (int, float))]
    submitted = len(scoped_predictions)
    successful = statuses["success"]
    return {
        "method": method,
        "stage": stage or "ALL",
        "submitted_cases": submitted,
        "successful_cases": successful,
        "error_cases": submitted - successful,
        "logical_model_calls": len(logical_calls),
        "actual_http_requests": len(scoped_events),
        "http_retry_requests": sum(1 for item in scoped_events if int(item.get("http_attempt", 1)) > 1),
        "requests_per_successful_case": (len(scoped_events) / successful) if successful else None,
        "usage_coverage": (len(usage_events) / len(scoped_events) * 100) if scoped_events else None,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else ["method", "stage"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(run_id: str, summaries: list[dict[str, Any]], event_count: int) -> str:
    overview = [row for row in summaries if row["stage"] == "ALL"]
    stage_rows = [row for row in summaries if row["stage"] != "ALL"]
    text = [
        "# Generation benchmark telemetry report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Observed VLLM HTTP attempts: **{event_count:,}**",
        "- Token totals use only provider `usage` fields. `N/A` means the response did not supply usage; no token estimate is used.",
        "- `HTTP retries` counts extra attempts within one logical model call; it does not count workflow loops such as G4 regeneration or G5 refinement.",
        "",
        "## Summary by method",
        "",
        "| Method | Cases (success/submitted) | Logical calls | HTTP requests | HTTP retries | Requests/success | Usage coverage | Input tokens | Output tokens | Total tokens | P50 / P95 request latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overview:
        rendered = {
            **row,
            "requests_display": _format_number(row["requests_per_successful_case"]),
            "coverage_display": (f"{row['usage_coverage']:.1f}%" if row["usage_coverage"] is not None else "N/A"),
            "input_tokens_display": _format_number(row["input_tokens"]),
            "output_tokens_display": _format_number(row["output_tokens"]),
            "total_tokens_display": _format_number(row["total_tokens"]),
            "p50_display": _format_number(row["p50_latency_ms"]),
            "p95_display": _format_number(row["p95_latency_ms"]),
        }
        text.append(
            "| {method} | {successful_cases:,}/{submitted_cases:,} | {logical_model_calls:,} | {actual_http_requests:,} | {http_retry_requests:,} | {requests_display} | {coverage_display} | {input_tokens_display} | {output_tokens_display} | {total_tokens_display} | {p50_display} / {p95_display} ms |".format(**rendered)
        )
    text.extend(["", "## Request stages", "", "| Method | Stage | Logical calls | HTTP requests | HTTP retries | Usage coverage | Total tokens | P50 / P95 latency |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in stage_rows:
        rendered = {
            **row,
            "coverage_display": (f"{row['usage_coverage']:.1f}%" if row["usage_coverage"] is not None else "N/A"),
            "total_tokens_display": _format_number(row["total_tokens"]),
            "p50_display": _format_number(row["p50_latency_ms"]),
            "p95_display": _format_number(row["p95_latency_ms"]),
        }
        text.append(
            "| {method} | {stage} | {logical_model_calls:,} | {actual_http_requests:,} | {http_retry_requests:,} | {coverage_display} | {total_tokens_display} | {p50_display} / {p95_display} ms |".format(**rendered)
        )
    return "\n".join(text) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a generation request/token benchmark report")
    parser.add_argument("--telemetry-dir", type=Path, required=True, help="One run directory under output/telemetry")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True, help="prediction JSONL files from the same run")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --telemetry-dir")
    args = parser.parse_args(argv)
    event_paths = sorted(args.telemetry_dir.glob("*_request_events.jsonl"))
    if not event_paths:
        raise SystemExit(f"No *_request_events.jsonl files in {args.telemetry_dir}")
    events = _rows(event_paths)
    if any(item.get("schema_version") != "1" for item in events):
        raise SystemExit("Unsupported telemetry schema version")
    run_ids = {item.get("run_id") for item in events}
    if len(run_ids) != 1 or not isinstance(next(iter(run_ids)), str):
        raise SystemExit("Telemetry directory must contain exactly one non-empty run_id")
    run_id = next(iter(run_ids))
    predictions = [item for item in _rows(args.predictions) if item.get("run_id") == run_id]
    if not predictions:
        raise SystemExit("No prediction rows match the telemetry run_id. Re-run generation with this telemetry version.")
    methods = sorted({str(item["method"]) for item in events})
    summaries: list[dict[str, Any]] = []
    for method in methods:
        summaries.append(_summary(events, predictions, method=method, stage=None))
        for stage in sorted({str(item["stage"]) for item in events if item.get("method") == method}):
            summaries.append(_summary(events, predictions, method=method, stage=stage))
    output_dir = args.output_dir or args.telemetry_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "generation_benchmark_report.md"
    csv_path = output_dir / "generation_benchmark_summary.csv"
    markdown_path.write_text(_markdown(run_id, summaries, len(events)), encoding="utf-8")
    _write_csv(csv_path, summaries)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
