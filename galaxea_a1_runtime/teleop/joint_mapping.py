"""Pure tracked configuration for the LeRobot A1 teleoperation processor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointMappingConfig:
    relative: bool
    input_degrees: bool
    scale: tuple[float, ...]
    sign: tuple[float, ...]
    bias_rad: tuple[float, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]

    def validate(self, dof: int) -> None:
        for name, values in (
            ("scale", self.scale),
            ("sign", self.sign),
            ("bias_rad", self.bias_rad),
            ("lower_limits", self.lower_limits),
            ("upper_limits", self.upper_limits),
        ):
            if len(values) != dof:
                raise ValueError(f"{name} expects {dof} values, got {len(values)}")
        for lo, hi in zip(self.lower_limits, self.upper_limits, strict=True):
            if lo > hi:
                raise ValueError(f"invalid joint limit: lower={lo} upper={hi}")
