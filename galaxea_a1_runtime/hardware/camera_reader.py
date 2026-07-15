"""Threaded latest-sample ownership independent of camera backends."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraSample:
    seq: int
    monotonic_s: float
    value: Any


class LatestCameraReader:
    """Continuously read one camera and expose the newest successful sample."""

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], Any | None],
        *,
        idle_sleep_s: float = 0.002,
    ):
        self.name = name
        self._read_fn = read_fn
        self._idle_sleep_s = idle_sleep_s
        self._lock = threading.Lock()
        self._latest: CameraSample | None = None
        self._exception: BaseException | None = None
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._run, name=f"{name}-camera-reader", daemon=True
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"{self.name} camera reader was already started")
        self._started = True
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def wait_stopped(self, *, timeout_s: float = 2.0) -> None:
        if not self._started:
            return
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(
                f"{self.name} camera reader did not stop within {timeout_s:.1f}s"
            )

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self.request_stop()
        self.wait_stopped(timeout_s=timeout_s)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def latest(self) -> CameraSample | None:
        with self._lock:
            return self._latest

    def latest_seq(self) -> int:
        latest = self.latest()
        return -1 if latest is None else latest.seq

    def frame_count(self) -> int:
        return self.latest_seq() + 1

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._exception

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            try:
                value = self._read_fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to owner thread.
                with self._lock:
                    self._exception = exc
                return
            if value is None:
                time.sleep(self._idle_sleep_s)
                continue
            with self._lock:
                self._latest = CameraSample(
                    seq=seq, monotonic_s=time.perf_counter(), value=value
                )
            seq += 1


def close_camera_resources(
    readers: Sequence[LatestCameraReader | None],
    cameras: Sequence[Any | None],
    *,
    timeout_s: float = 2.0,
) -> None:
    """Stop readers and cameras as one cleanup unit, surfacing every failure."""

    errors: list[BaseException] = []
    for reader in readers:
        if reader is not None:
            reader.request_stop()
    for camera in cameras:
        if camera is None:
            continue
        try:
            camera.close()
        except BaseException as exc:  # noqa: BLE001 - cleanup must continue.
            errors.append(exc)
    for reader in readers:
        if reader is None:
            continue
        try:
            reader.wait_stopped(timeout_s=timeout_s)
        except BaseException as exc:  # noqa: BLE001 - cleanup must continue.
            errors.append(exc)
    if errors:
        summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        raise RuntimeError(f"camera cleanup failed: {summary}") from errors[0]
