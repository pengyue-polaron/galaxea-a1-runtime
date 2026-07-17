"""Reusable feedback caches for A1 joint-space applications."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

from galaxea_a1_runtime.gripper import normalize_stroke
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


def ordered_joint_positions(
    msg: Any,
    ordered_names: Sequence[str],
    *,
    label: str,
    allow_unnamed: bool = True,
) -> tuple[float, ...]:
    """Decode a JointState-like message without silently changing joint order."""

    expected = tuple(str(name) for name in ordered_names)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError(f"{label} expected joint names must be non-empty and unique")

    names = tuple(str(name) for name in getattr(msg, "name", ()))
    raw_values = tuple(getattr(msg, "position", ()))
    if len(raw_values) < len(expected):
        raise ValueError(
            f"{label} has {len(raw_values)} positions, need {len(expected)}"
        )

    if names:
        if len(names) != len(raw_values):
            raise ValueError(
                f"{label} has {len(names)} names but {len(raw_values)} positions"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{label} contains duplicate joint names")
        by_name = dict(zip(names, raw_values, strict=True))
        missing = tuple(name for name in expected if name not in by_name)
        if missing:
            raise ValueError(f"{label} is missing expected joints: {list(missing)}")
        values = tuple(float(by_name[name]) for name in expected)
    else:
        if not allow_unnamed:
            raise ValueError(f"{label} must include joint names")
        values = tuple(float(value) for value in raw_values[: len(expected)])

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains non-finite joint positions")
    return values


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
        return ordered_joint_positions(
            msg,
            self.ordered_names,
            label="A1 joint feedback",
        )


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

    def max_error(
        self,
        target: Sequence[float],
        dof: int,
        *,
        max_age_s: float | None = None,
    ) -> float | None:
        msg = self.cache.get(max_age_s=max_age_s)
        if msg is None or len(getattr(msg, "p_des", ())) < dof:
            return None
        staged = tuple(float(value) for value in msg.p_des[:dof])
        if not all(math.isfinite(value) for value in staged):
            return None
        return max(abs(staged[index] - float(target[index])) for index in range(dof))


def wait_for_staged_joint_alignment(
    monitor: StagedCommandMonitor,
    target: Sequence[float],
    *,
    dof: int,
    timeout_s: float,
    max_age_s: float,
    tolerance_rad: float,
    is_shutdown,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> float:
    """Wait until jointTracker has produced a fresh command aligned to a hold."""

    deadline = monotonic() + timeout_s
    last_error: float | None = None
    while not is_shutdown() and monotonic() < deadline:
        last_error = monitor.max_error(target, dof, max_age_s=max_age_s)
        if last_error is not None and last_error <= tolerance_rad:
            return last_error
        sleep(0.02)
    detail = "no complete fresh staged command"
    if last_error is not None:
        detail = f"last max error={last_error:.6f}rad"
    raise RuntimeError(
        "jointTracker did not stage the initial hold within "
        f"{timeout_s:.1f}s ({detail})"
    )
