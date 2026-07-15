"""Pure SO leader to A1 joint mapping helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


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


def detect_leader_joint_keys(action: Mapping[str, float], dof: int) -> tuple[str, ...]:
    current = tuple(f"joint{i}.pos" for i in range(dof))
    if all(key in action for key in current):
        return current

    raise RuntimeError(
        f"Could not detect {dof} leader joint keys from action keys: {sorted(action)}. "
        "Expected A1 SO leader keys joint0.pos..joint5.pos."
    )


def map_leader_joints_to_a1(
    *,
    leader_now: Sequence[float],
    leader_start: Sequence[float],
    a1_start: Sequence[float],
    config: JointMappingConfig,
) -> tuple[float, ...]:
    dof = len(a1_start)
    config.validate(dof)
    if len(leader_now) != dof or len(leader_start) != dof:
        raise ValueError("leader and A1 joint vectors must have the same length")

    if config.relative:
        values = tuple(
            float(leader_now[i]) - float(leader_start[i]) for i in range(dof)
        )
        if config.input_degrees:
            values = tuple(math.radians(value) for value in values)
        target = tuple(
            float(a1_start[i])
            + config.sign[i] * config.scale[i] * values[i]
            + config.bias_rad[i]
            for i in range(dof)
        )
    else:
        values = tuple(float(value) for value in leader_now)
        if config.input_degrees:
            values = tuple(math.radians(value) for value in values)
        target = tuple(
            config.sign[i] * config.scale[i] * values[i] + config.bias_rad[i]
            for i in range(dof)
        )

    return tuple(
        min(config.upper_limits[i], max(config.lower_limits[i], target[i]))
        for i in range(dof)
    )
