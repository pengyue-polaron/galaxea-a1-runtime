"""Thread-safe reset progress presentation."""

from __future__ import annotations

import sys
import threading

from galaxea_a1_runtime.console import Tone, emit, style


class ResetProgress:
    def __init__(self, devices: tuple[str, ...]):
        self.devices = devices
        self.values = {device: 0 for device in devices}
        self.reported = {device: -1 for device in devices}
        self.lock = threading.Lock()
        self.interactive = sys.stdout.isatty()

    def update(self, device: str, percent: float) -> None:
        value = max(0, min(100, int(round(percent))))
        with self.lock:
            if value == self.reported[device]:
                return
            self.values[device] = value
            self.reported[device] = value
            if self.interactive:
                status = " | ".join(
                    f"{name} {self.values[name]:3d}%" for name in self.devices
                )
                prefix = style("Reset", Tone.STEP)
                print(f"\r\033[2K{prefix}  {status}", end="", flush=True)
            elif value in {0, 25, 50, 75, 100}:
                emit("STEP", f"Reset {device}: {value}%")

    def finish(self, *, success: bool) -> None:
        if self.interactive:
            print("\r\033[2K", end="")
        emit(
            "PASS" if success else "FAIL",
            "Reset complete" if success else "Reset failed",
            stream=sys.stdout if success else sys.stderr,
        )
