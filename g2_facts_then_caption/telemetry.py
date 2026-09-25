"""Thread-safe, terminal-only counters for one G2 prediction case."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CallContext:
    sample_id: str
    stage: str


class CaseMetrics:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self._lock = threading.Lock()
        self._data = {"input_token": 0, "output_token": 0, "num_request": 0, "num_retry": 0, "num_error_request": 0, "usage_count": 0}

    def record(self, context: CallContext, *, http_attempt: int, outcome: str, input_token: int | None, output_token: int | None) -> None:
        with self._lock:
            self._data["num_request"] += 1
            self._data["num_retry"] += int(http_attempt > 1)
            self._data["num_error_request"] += int(outcome != "success")
            if input_token is not None or output_token is not None:
                self._data["usage_count"] += 1
                self._data["input_token"] += input_token or 0
                self._data["output_token"] += output_token or 0
            print(f"[g2] sample={context.sample_id} stage={context.stage} request={self._data['num_request']} outcome={outcome} input_token={input_token} output_token={output_token}", flush=True)

    def summary(self) -> dict[str, int | None]:
        with self._lock:
            return {"input_token": self._data["input_token"] if self._data["usage_count"] else None, "output_token": self._data["output_token"] if self._data["usage_count"] else None, "num_request": self._data["num_request"], "num_retry": self._data["num_retry"], "num_error_request": self._data["num_error_request"]}
