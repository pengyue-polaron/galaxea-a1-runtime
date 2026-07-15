"""Pure safety helpers for the Galaxea A1 runtime.

This module must stay free of ROS imports and hardware side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .constants import (
    ARM_JOINT_COUNT,
    IDLE_TIMEOUT_CODE,
)


@dataclass(frozen=True)
class RelayInputs:
    """Snapshot used to decide whether the relay may forward one command."""

    enabled: bool
    joint_age: float
    source_age: float
    status_age: float
    joint_count: int
    source_count: int
    motor_error_codes: tuple[int, ...]


@dataclass(frozen=True)
class SafetyDecision:
    """Result of a fail-closed safety check."""

    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> "SafetyDecision":
        return cls(True, None)

    @classmethod
    def block(cls, reason: str) -> "SafetyDecision":
        return cls(False, reason)


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned EEF workspace bounds in meters."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def clamp(self, xyz: Sequence[float]) -> tuple[float, float, float]:
        if len(xyz) != 3:
            raise ValueError(f"workspace clamp expects 3 values, got {len(xyz)}")
        return (
            clamp_float(xyz[0], self.x[0], self.x[1]),
            clamp_float(xyz[1], self.y[0], self.y[1]),
            clamp_float(xyz[2], self.z[0], self.z[1]),
        )


def clamp_float(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError(f"invalid bounds: lower={lower} upper={upper}")
    if not isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    return min(max(value, lower), upper)


def validate_relay_inputs(
    inputs: RelayInputs,
    *,
    arm_joints: int = ARM_JOINT_COUNT,
    max_input_age: float,
    max_status_age: float,
) -> SafetyDecision:
    """Return a fail-closed decision for one staged joint command."""

    reason = relay_block_reason(
        inputs,
        arm_joints=arm_joints,
        max_input_age=max_input_age,
        max_status_age=max_status_age,
    )
    if reason is not None:
        return SafetyDecision.block(reason)
    return SafetyDecision.allow()


def relay_block_reason(
    inputs: RelayInputs,
    *,
    arm_joints: int = ARM_JOINT_COUNT,
    max_input_age: float,
    max_status_age: float,
) -> str | None:
    if not inputs.enabled:
        return "locked"
    if arm_joints <= 0:
        return f"invalid arm joint count: {arm_joints}"
    if max_input_age <= 0:
        return f"invalid max input age: {max_input_age}"
    if max_status_age <= 0:
        return f"invalid max status age: {max_status_age}"
    if inputs.joint_count < arm_joints:
        return f"joint feedback has {inputs.joint_count} positions, need {arm_joints}"
    if inputs.source_count < arm_joints:
        return f"tracker command has {inputs.source_count} positions, need {arm_joints}"
    if inputs.joint_age > max_input_age:
        return f"joint feedback stale ({inputs.joint_age:.3f}s)"
    if inputs.source_age > max_input_age:
        return f"tracker command stale ({inputs.source_age:.3f}s)"
    if inputs.status_age > max_status_age:
        return f"motor status stale ({inputs.status_age:.3f}s)"
    if len(inputs.motor_error_codes) < arm_joints:
        return f"motor status has {len(inputs.motor_error_codes)} entries, need {arm_joints}"

    bad = [
        (i + 1, code)
        for i, code in enumerate(inputs.motor_error_codes[:arm_joints])
        if code not in (0, IDLE_TIMEOUT_CODE)
    ]
    if bad:
        return "motor errors: " + ", ".join(f"J{joint}={code}" for joint, code in bad)
    return None


def validate_initial_alignment(
    current: Sequence[float],
    raw: Sequence[float],
    *,
    max_abs_error: float,
) -> None:
    """Reject a large first tracker jump without modifying the command."""

    _require_same_length(current, raw, "current", "raw")
    if max_abs_error < 0:
        raise ValueError(f"max_abs_error must be non-negative, got {max_abs_error}")
    error = tuple(raw[i] - current[i] for i in range(len(current)))
    if error and max(abs(v) for v in error) > max_abs_error:
        raise ValueError(
            f"initial command error exceeds {max_abs_error}: {list(error)}"
        )


def gripper_stroke_block_reason(
    stroke_mm: float, *, minimum_mm: float, maximum_mm: float
) -> str | None:
    """Validate a physical gripper target without clamping or rewriting it."""

    if not isfinite(minimum_mm) or not isfinite(maximum_mm) or minimum_mm >= maximum_mm:
        return f"invalid gripper range [{minimum_mm}, {maximum_mm}]mm"
    if not isfinite(stroke_mm):
        return f"gripper target is not finite: {stroke_mm!r}"
    if stroke_mm < minimum_mm or stroke_mm > maximum_mm:
        return (
            f"gripper target {stroke_mm:g}mm is outside "
            f"[{minimum_mm:g}, {maximum_mm:g}]mm"
        )
    return None


def actuator_error_block_reason(
    motor_error_codes: Sequence[int],
    *,
    index: int,
    label: str,
    ignored_mask: int = 0,
) -> str | None:
    """Return a fault reason after removing explicitly accepted status bits."""

    if index < 0:
        return f"invalid {label} motor index: {index}"
    if len(motor_error_codes) <= index:
        return f"motor status has {len(motor_error_codes)} entries, need {index + 1} for {label}"
    if ignored_mask < 0:
        return f"invalid {label} ignored motor error mask: {ignored_mask}"
    code = int(motor_error_codes[index])
    remaining = code & ~(IDLE_TIMEOUT_CODE | int(ignored_mask))
    if remaining:
        return f"{label} motor error: {code}"
    return None


def require_finite_vector(
    values: Sequence[float], *, count: int, label: str
) -> tuple[float, ...]:
    """Validate the required prefix of one hardware vector without rewriting it."""

    if count <= 0:
        raise ValueError(f"{label} required count must be positive, got {count}")
    if len(values) < count:
        raise ValueError(f"{label} has {len(values)} values, need {count}")
    prefix = tuple(float(value) for value in values[:count])
    if not all(isfinite(value) for value in prefix):
        raise ValueError(f"{label} contains non-finite values")
    return prefix


def validate_arm_control_command(
    *,
    p_des: Sequence[float],
    v_des: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    t_ff: Sequence[float],
    mode: int,
    arm_joints: int,
    allowed_modes: Sequence[int],
) -> None:
    """Validate every hardware-affecting field of one driver command."""

    vectors = {
        "p_des": p_des,
        "v_des": v_des,
        "kp": kp,
        "kd": kd,
        "t_ff": t_ff,
    }
    for label, values in vectors.items():
        if len(values) != arm_joints:
            raise ValueError(
                f"tracker command {label} has {len(values)} values, "
                f"need exactly {arm_joints}"
            )
        require_finite_vector(
            values,
            count=arm_joints,
            label=f"tracker command {label}",
        )
    if any(float(value) < 0.0 for value in kp):
        raise ValueError("tracker command kp contains negative gains")
    if any(float(value) < 0.0 for value in kd):
        raise ValueError("tracker command kd contains negative gains")
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError(f"tracker command mode is not an integer: {mode!r}")
    if mode not in allowed_modes:
        raise ValueError(
            f"tracker command mode {mode} is not allowed; "
            f"expected one of {list(allowed_modes)}"
        )


def clamp_eef_delta(
    delta: Sequence[float],
    *,
    max_translation: float,
    max_rotation: float | None = None,
) -> tuple[float, ...]:
    """Clamp an EEF delta action.

    Supported shapes are translation-only `[dx, dy, dz]`, translation plus
    gripper `[dx, dy, dz, gripper]`, or full pose delta plus gripper
    `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.
    """

    if len(delta) not in (3, 4, 7):
        raise ValueError(f"unsupported EEF delta length: {len(delta)}")
    if max_translation < 0:
        raise ValueError(f"max_translation must be non-negative, got {max_translation}")
    if len(delta) == 7:
        if max_rotation is None:
            raise ValueError("max_rotation is required for a full EEF delta")
        if max_rotation < 0:
            raise ValueError(f"max_rotation must be non-negative, got {max_rotation}")
    elif max_rotation is not None:
        raise ValueError("max_rotation is only valid for a full EEF delta")

    values = [float(v) for v in delta]
    clamped = [
        clamp_float(values[i], -max_translation, max_translation) for i in range(3)
    ]
    if len(values) == 7:
        clamped.extend(
            clamp_float(values[i], -max_rotation, max_rotation) for i in range(3, 6)
        )
        clamped.append(clamp_float(values[6], 0.0, 1.0))
    elif len(values) == 4:
        clamped.append(clamp_float(values[3], 0.0, 1.0))
    return tuple(clamped)


def _require_same_length(
    left: Sequence[float],
    right: Sequence[float],
    left_name: str,
    right_name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(
            f"{left_name} and {right_name} length mismatch: {len(left)} != {len(right)}"
        )
