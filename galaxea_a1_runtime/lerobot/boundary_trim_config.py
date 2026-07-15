"""Pure configuration values for conservative episode-boundary trimming."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryTrimConfig:
    """Tracked policy for removing only stationary episode boundaries."""

    enabled: bool
    anchor_window_s: float
    joint_deadband_rad: float
    gripper_deadband: float
    confirm_frames: int
    pre_roll_s: float
    post_roll_s: float
    max_trim_fraction: float
    min_kept_duration_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("trim.enabled must be boolean")
        positive = {
            "anchor_window_s": self.anchor_window_s,
            "joint_deadband_rad": self.joint_deadband_rad,
            "gripper_deadband": self.gripper_deadband,
            "min_kept_duration_s": self.min_kept_duration_s,
        }
        for key, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"trim.{key} must be finite and positive")
        if isinstance(self.confirm_frames, bool) or self.confirm_frames <= 0:
            raise ValueError("trim.confirm_frames must be positive")
        if (
            not math.isfinite(self.pre_roll_s)
            or not math.isfinite(self.post_roll_s)
            or self.pre_roll_s < 0
            or self.post_roll_s < 0
        ):
            raise ValueError("trim pre/post roll must be finite and non-negative")
        if (
            not math.isfinite(self.max_trim_fraction)
            or not 0 < self.max_trim_fraction < 1
        ):
            raise ValueError("trim.max_trim_fraction must be between 0 and 1")
