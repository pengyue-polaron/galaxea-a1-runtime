"""Thread-safe cache for timestamped hardware messages."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar


MessageT = TypeVar("MessageT")


class LatestMessageCache(Generic[MessageT]):
    """Keep the latest message and reject it after a monotonic deadline."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._msg: MessageT | None = None
        self._updated_monotonic: float | None = None

    def set(self, msg: MessageT) -> None:
        with self._lock:
            self._msg = msg
            self._updated_monotonic = self._clock()

    callback = set

    def snapshot(self) -> tuple[MessageT | None, float | None]:
        """Return the value and its monotonic update time atomically."""
        with self._lock:
            return self._msg, self._updated_monotonic

    def get(self, *, max_age_s: float | None = None) -> MessageT | None:
        if max_age_s is not None and max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        with self._lock:
            if max_age_s is not None and (
                self._updated_monotonic is None
                or self._clock() - self._updated_monotonic > max_age_s
            ):
                return None
            return self._msg
