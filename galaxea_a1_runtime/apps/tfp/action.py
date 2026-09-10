"""Pure TFP action-to-A1 command mapping."""

from __future__ import annotations

import math
from collections.abc import Sequence

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES_RAD


def tfp_action_to_command(
    action: Sequence[object], system: SystemConfig
) -> dict[str, float]:
    if len(action) != len(JOINT_ACTION_NAMES_RAD):
        raise ValueError(
            f"TFP action must contain {len(JOINT_ACTION_NAMES_RAD)} values, got {len(action)}"
        )
    values = tuple(float(value) for value in action)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("TFP action must contain only finite values")
    for name, value, lower, upper in zip(
        JOINT_ACTION_NAMES_RAD[:-1],
        values[:-1],
        system.joint_safety.lower_limits,
        system.joint_safety.upper_limits,
        strict=True,
    ):
        if not lower <= value <= upper:
            raise ValueError(f"TFP {name}={value:g} is outside [{lower:g}, {upper:g}]")
    if not 0.0 <= values[-1] <= 1.0:
        raise ValueError("TFP gripper_normalized must be in [0, 1]")
    return dict(zip(JOINT_ACTION_NAMES_RAD, values, strict=True))
