"""Thread-safe rolling request limiter for one G1 process."""

from collections import deque
from threading import Condition
from time import monotonic


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least one")
        self._limit = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._condition = Condition()

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = monotonic()
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return
                self._condition.wait(timeout=max(0.001, self._timestamps[0] + 60.0 - now))
