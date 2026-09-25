"""Thread-safe rolling request limiter for one G3 process."""

from collections import deque
from threading import Condition
from time import monotonic


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least one")
        self._limit, self._timestamps, self._condition = requests_per_minute, deque(), Condition()

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = monotonic()
                while self._timestamps and self._timestamps[0] <= now - 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return
                self._condition.wait(timeout=max(0.001, self._timestamps[0] + 60.0 - now))
