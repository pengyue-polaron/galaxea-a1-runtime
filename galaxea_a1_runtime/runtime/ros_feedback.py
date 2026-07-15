"""Reusable feedback caches for A1 joint-space applications."""

from __future__ import annotations

from typing import Any, Sequence

from galaxea_a1_runtime.gripper import normalize_stroke
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


class A1JointStateCache:
    def __init__(self, ordered_names: tuple[str, ...]):
        self.ordered_names = ordered_names
        self.cache: LatestMessageCache[Any] = LatestMessageCache()

    def callback(self, msg: Any) -> None:
        self.cache.set(msg)

    def positions(self, *, max_age_s: float | None = None) -> tuple[float, ...] | None:
        msg = self.cache.get(max_age_s=max_age_s)
        if msg is None:
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
        self.cache: LatestMessageCache[Any] = LatestMessageCache()

    def callback(self, msg: Any) -> None:
        self.cache.set(msg)

    def normalized(
        self,
        *,
        max_age_s: float,
        stroke_min_mm: float,
        stroke_max_mm: float,
    ) -> float | None:
        msg = self.cache.get(max_age_s=max_age_s)
        values = [] if msg is None else list(getattr(msg, "position", []))
        if not values:
            return None
        return normalize_stroke(
            float(values[0]),
            stroke_min_mm=stroke_min_mm,
            stroke_max_mm=stroke_max_mm,
        )


class StagedCommandMonitor:
    def __init__(self):
        self.cache: LatestMessageCache[Any] = LatestMessageCache()

    def callback(self, msg: Any) -> None:
        self.cache.set(msg)

    def max_error(self, target: Sequence[float], dof: int) -> float | None:
        msg = self.cache.get()
        if msg is None or len(getattr(msg, "p_des", ())) < dof:
            return None
        staged = tuple(float(value) for value in msg.p_des[:dof])
        return max(abs(staged[index] - float(target[index])) for index in range(dof))
