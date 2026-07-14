"""ROS feedback caches used by the ACT joint bridge."""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence

from galaxea_a1_runtime.apps.eef_bridge import (
    RelayStatus,
    decode_relay_status,
    relay_state_summary,
    relay_status_is_fresh,
)
from galaxea_a1_runtime.gripper import normalize_stroke
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from std_msgs.msg import String


class LatestCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._value: Any | None = None
        self._updated_monotonic: float | None = None

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._updated_monotonic = time.monotonic()

    def get(self) -> tuple[Any | None, float | None]:
        with self._lock:
            return self._value, self._updated_monotonic


class A1JointStateCache:
    def __init__(self, ordered_names: tuple[str, ...]):
        self.ordered_names = ordered_names
        self.cache = LatestCache()

    def callback(self, msg: JointState) -> None:
        self.cache.set(msg)

    def positions(self, *, max_age_s: float) -> tuple[float, ...] | None:
        msg, updated = self.cache.get()
        if msg is None or updated is None or time.monotonic() - updated > max_age_s:
            return None
        names = list(getattr(msg, "name", []))
        values = list(getattr(msg, "position", []))
        if len(values) < len(self.ordered_names):
            return None
        name_to_idx = {name: index for index, name in enumerate(names)}
        if names and all(name in name_to_idx for name in self.ordered_names):
            indices = [name_to_idx[name] for name in self.ordered_names]
            if all(index < len(values) for index in indices):
                return tuple(float(values[index]) for index in indices)
        return tuple(float(value) for value in values[: len(self.ordered_names)])


class GripperFeedbackCache:
    def __init__(self):
        self.cache = LatestCache()

    def callback(self, msg: JointState) -> None:
        self.cache.set(msg)

    def normalized(
        self,
        *,
        max_age_s: float,
        stroke_min_mm: float,
        stroke_max_mm: float,
    ) -> float | None:
        msg, updated = self.cache.get()
        if msg is None or updated is None or time.monotonic() - updated > max_age_s:
            return None
        values = list(getattr(msg, "position", []))
        if not values:
            return None
        return normalize_stroke(
            float(values[0]),
            stroke_min_mm=stroke_min_mm,
            stroke_max_mm=stroke_max_mm,
        )


class RelayMonitor:
    def __init__(self, max_status_age_s: float):
        self.max_status_age_s = max_status_age_s
        self.cache = LatestCache()

    def callback(self, msg: String) -> None:
        self.cache.set(decode_relay_status(msg.data))

    def status(self) -> tuple[RelayStatus | None, float | None]:
        value, updated = self.cache.get()
        return value, updated

    def summary(self) -> str:
        status, updated = self.status()
        return relay_state_summary(status, updated, max_age_s=self.max_status_age_s)

    def is_active(self) -> bool:
        status, updated = self.status()
        return (
            relay_status_is_fresh(updated, max_age_s=self.max_status_age_s)
            and (status or RelayStatus("UNKNOWN")).state == "ACTIVE"
        )


class StagedCommandMonitor:
    def __init__(self):
        self.cache = LatestCache()

    def callback(self, msg: arm_control) -> None:
        self.cache.set(msg)

    def max_error(self, target: Sequence[float], dof: int) -> float | None:
        msg, _ = self.cache.get()
        if msg is None or len(getattr(msg, "p_des", ())) < dof:
            return None
        staged = tuple(float(value) for value in msg.p_des[:dof])
        return max(abs(staged[index] - float(target[index])) for index in range(dof))
