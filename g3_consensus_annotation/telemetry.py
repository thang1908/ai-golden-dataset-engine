"""Local, privacy-safe per-request telemetry for G3."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CallContext:
    run_id: str
    method: str
    sample_id: str
    stage: str
    logical_call_id: str


class SampleCallFactory:
    def __init__(self, *, run_id: str, method: str, sample_id: str) -> None:
        self._run_id, self._method, self._sample_id = run_id, method, sample_id
        self._counts: dict[str, int] = {}

    def next(self, stage: str) -> CallContext:
        count = self._counts.get(stage, 0) + 1
        self._counts[stage] = count
        return CallContext(
            run_id=self._run_id,
            method=self._method,
            sample_id=self._sample_id,
            stage=stage,
            logical_call_id=f"{self._method}:{self._sample_id}:{stage}:{count}",
        )


class TelemetryWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._sample_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(
        self,
        context: CallContext,
        *,
        http_attempt: int,
        outcome: str,
        latency_ms: float,
        http_status: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        retry_reason: str | None,
        has_image: bool,
        model_id: str,
    ) -> None:
        event: dict[str, Any] = {
            "schema_version": "1",
            "run_id": context.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "method": context.method,
            "sample_id": context.sample_id,
            "stage": context.stage,
            "logical_call_id": context.logical_call_id,
            "http_attempt": http_attempt,
            "model_id": model_id,
            "has_image": has_image,
            "http_status": http_status,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "usage_available": any(value is not None for value in (input_tokens, output_tokens, total_tokens)),
            "retry_reason": retry_reason,
        }
        with self._lock:
            self._stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._stream.flush()
            key = (context.run_id, context.method, context.sample_id)
            metrics = self._sample_metrics.setdefault(
                key,
                {
                    "logical_call_ids": set(), "http_request_count": 0, "http_retry_count": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "usage_event_count": 0,
                },
            )
            metrics["logical_call_ids"].add(context.logical_call_id)
            metrics["http_request_count"] += 1
            metrics["http_retry_count"] += int(http_attempt > 1)
            if event["usage_available"]:
                metrics["usage_event_count"] += 1
                for field in ("input_tokens", "output_tokens", "total_tokens"):
                    if isinstance(event[field], int):
                        metrics[field] += event[field]

    def sample_summary(self, *, run_id: str, method: str, sample_id: str) -> dict[str, Any]:
        with self._lock:
            metrics = self._sample_metrics.get((run_id, method, sample_id))
            if metrics is None:
                return {"logical_call_count": 0, "http_request_count": 0, "http_retry_count": 0, "input_tokens": None, "output_tokens": None, "total_tokens": None, "token_usage_coverage_percent": None}
            request_count = metrics["http_request_count"]
            return {
                "logical_call_count": len(metrics["logical_call_ids"]),
                "http_request_count": request_count,
                "http_retry_count": metrics["http_retry_count"],
                "input_tokens": metrics["input_tokens"] if metrics["usage_event_count"] else None,
                "output_tokens": metrics["output_tokens"] if metrics["usage_event_count"] else None,
                "total_tokens": metrics["total_tokens"] if metrics["usage_event_count"] else None,
                "token_usage_coverage_percent": round(metrics["usage_event_count"] / request_count * 100, 1) if request_count else None,
            }

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()

    def record(
        self,
        context: CallContext,
        *,
        http_attempt: int,
        model_id: str,
        has_image: bool,
        outcome: str,
        latency_ms: float,
        http_status: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        retry_reason: str | None,
    ) -> None:
        self.append(
            context, http_attempt=http_attempt, model_id=model_id, has_image=has_image,
            outcome=outcome, latency_ms=latency_ms, http_status=http_status,
            input_tokens=prompt_tokens, output_tokens=completion_tokens,
            total_tokens=total_tokens, retry_reason=retry_reason,
        )


def default_run_id(method: str) -> str:
    return f"{method}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
