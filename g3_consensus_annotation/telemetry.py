"""Per-case request and token counters printed safely to the terminal."""

from __future__ import annotations

import threading
from dataclasses import dataclass


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
        return CallContext(self._run_id, self._method, self._sample_id, stage, f"{self._method}:{self._sample_id}:{stage}:{count}")


class TelemetryWriter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[tuple[str, str, str], dict[str, object]] = {}

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def record(self, context: CallContext, *, http_attempt: int, model_id: str, has_image: bool, outcome: str, latency_ms: float, http_status: int | None, prompt_tokens: int | None, completion_tokens: int | None, total_tokens: int | None, retry_reason: str | None) -> None:
        del model_id, has_image, latency_ms, http_status, total_tokens, retry_reason
        key = (context.run_id, context.method, context.sample_id)
        with self._lock:
            metrics = self._metrics.setdefault(key, {"logical_call_ids": set(), "num_request": 0, "num_retry": 0, "num_error_request": 0, "input_token": 0, "output_token": 0, "usage_count": 0})
            logical_call_ids = metrics["logical_call_ids"]
            assert isinstance(logical_call_ids, set)
            logical_call_ids.add(context.logical_call_id)
            for name, increment in (("num_request", 1), ("num_retry", int(http_attempt > 1)), ("num_error_request", int(outcome != "success"))):
                metrics[name] = int(metrics[name]) + increment
            if prompt_tokens is not None or completion_tokens is not None:
                metrics["usage_count"] = int(metrics["usage_count"]) + 1
                metrics["input_token"] = int(metrics["input_token"]) + (prompt_tokens or 0)
                metrics["output_token"] = int(metrics["output_token"]) + (completion_tokens or 0)
            print(f"[{context.method}] sample={context.sample_id} stage={context.stage} request={metrics['num_request']} outcome={outcome} input_token={prompt_tokens} output_token={completion_tokens}", flush=True)

    def sample_summary(self, *, run_id: str, method: str, sample_id: str) -> dict[str, int | None]:
        with self._lock:
            metrics = self._metrics.get((run_id, method, sample_id))
            if metrics is None:
                return {"input_token": None, "output_token": None, "num_request": 0, "num_retry": 0, "num_error_request": 0}
            has_usage = bool(metrics["usage_count"])
            return {"input_token": int(metrics["input_token"]) if has_usage else None, "output_token": int(metrics["output_token"]) if has_usage else None, "num_request": int(metrics["num_request"]), "num_retry": int(metrics["num_retry"]), "num_error_request": int(metrics["num_error_request"])}


def default_run_id(method: str) -> str:
    return f"{method}_run"
