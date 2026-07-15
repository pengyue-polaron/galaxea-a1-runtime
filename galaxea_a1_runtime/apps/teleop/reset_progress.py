"""Thread-safe reset progress presentation."""

from __future__ import annotations

import os
import sys
import threading


class ResetProgress:
    def __init__(self, devices: tuple[str, ...]):
        self.devices = devices
        self.values = {device: 0 for device in devices}
        self.reported = {device: -1 for device in devices}
        self.lock = threading.Lock()
        self.interactive = sys.stdout.isatty()
        self.color = self.interactive and not os.environ.get("NO_COLOR")

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
                prefix = "\033[1;36mReset\033[0m" if self.color else "Reset"
                print(f"\r\033[2K{prefix}  {status}", end="", flush=True)
            elif value in {0, 25, 50, 75, 100}:
                print(f"[Reset] {device} {value}%", flush=True)

    def finish(self, *, success: bool) -> None:
        if self.interactive:
            print("\r\033[2K", end="")
        text = "[Reset] Complete" if success else "[Reset] Failed"
        if self.color:
            code = "\033[1;32m" if success else "\033[1;31m"
            text = f"{code}{text}\033[0m"
        print(text, flush=True)
