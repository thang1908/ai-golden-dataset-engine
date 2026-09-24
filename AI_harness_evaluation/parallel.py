from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TypeVar

Sample = TypeVar("Sample")
Result = TypeVar("Result")


def bounded_parallel(
    samples: Iterable[Sample],
    worker: Callable[[Sample], Result],
    workers: int,
) -> Iterator[Result]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    iterator = iter(samples)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit_next() -> bool:
            try:
                sample = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(worker, sample)] = sample
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                yield future.result()
                submit_next()
