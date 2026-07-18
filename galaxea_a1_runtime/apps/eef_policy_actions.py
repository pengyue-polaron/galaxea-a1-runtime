"""Pure EEF policy action transforms shared by model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.gripper import denormalize_stroke, normalize_stroke


@dataclass(frozen=True, kw_only=True)
class EefActionTransformConfig:
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    min_quat_norm: float
    gripper_stroke_min: float
    gripper_stroke_max: float


def build_action_transform_config(
    *,
    system: SystemConfig,
) -> EefActionTransformConfig:
    """Derive the pure action transform from its exclusive config owners."""

    return EefActionTransformConfig(
        xyz_min=system.eef.xyz_min,
        xyz_max=system.eef.xyz_max,
        min_quat_norm=system.eef.min_quat_norm,
        gripper_stroke_min=system.gripper.stroke_min_mm,
        gripper_stroke_max=system.gripper.stroke_max_mm,
    )


def normalize_condition_action(
    action8: Sequence[float], config: EefActionTransformConfig
) -> np.ndarray:
    """Normalize observed EE state without applying command workspace bounds."""

    action = _as_action8(action8, label="EE state condition")
    action[3:7] = normalize_quat(
        action[3:7], min_norm=config.min_quat_norm, label="EE state condition"
    )
    action[7] = _continuous_gripper(action[7])
    return action


def validate_policy_action(
    raw8: Sequence[float],
    config: EefActionTransformConfig,
) -> np.ndarray:
    """Normalize a model action and reject values outside explicit bounds."""

    action = _as_action8(raw8, label="EEF policy action")
    _validate_xyz(action[:3], config)
    action[3:7] = normalize_quat(
        action[3:7], min_norm=config.min_quat_norm, label="EEF policy action"
    )
    action[7] = _continuous_gripper(action[7])
    return action


def gripper_norm_from_stroke(
    stroke_mm: float, config: EefActionTransformConfig
) -> float:
    return normalize_stroke(
        stroke_mm,
        stroke_min_mm=config.gripper_stroke_min,
        stroke_max_mm=config.gripper_stroke_max,
    )


def gripper_stroke_from_norm(norm: float, config: EefActionTransformConfig) -> float:
    return denormalize_stroke(
        norm,
        stroke_min_mm=config.gripper_stroke_min,
        stroke_max_mm=config.gripper_stroke_max,
    )


def relative_action_to_absolute(
    relative8: Sequence[float],
    origin_pose7: Sequence[float],
    *,
    min_quat_norm: float,
) -> np.ndarray:
    """Compose an episode-relative xyz+xyzw action onto the episode origin."""

    relative = _as_action8(relative8, label="relative EEF policy action")
    origin = _as_pose7(origin_pose7, label="episode origin")
    relative_quat = normalize_quat(
        relative[3:7], min_norm=min_quat_norm, label="relative quaternion"
    )
    origin_quat = normalize_quat(
        origin[3:7], min_norm=min_quat_norm, label="origin quaternion"
    )
    absolute = relative.copy()
    absolute[:3] = origin[:3] + relative[:3]
    absolute[3:7] = normalize_quat(
        _quat_multiply(origin_quat, relative_quat),
        min_norm=min_quat_norm,
        label="absolute quaternion",
    )
    return absolute


def absolute_action_to_relative(
    absolute8: Sequence[float],
    origin_pose7: Sequence[float],
    *,
    min_quat_norm: float,
) -> np.ndarray:
    """Express an absolute xyz+xyzw action relative to the episode origin."""

    absolute = _as_action8(absolute8, label="absolute A1 action")
    origin = _as_pose7(origin_pose7, label="episode origin")
    absolute_quat = normalize_quat(
        absolute[3:7], min_norm=min_quat_norm, label="absolute quaternion"
    )
    origin_quat = normalize_quat(
        origin[3:7], min_norm=min_quat_norm, label="origin quaternion"
    )
    relative = absolute.copy()
    relative[:3] = absolute[:3] - origin[:3]
    relative[3:7] = normalize_quat(
        _quat_multiply(_quat_inverse(origin_quat), absolute_quat),
        min_norm=min_quat_norm,
        label="relative quaternion",
    )
    return relative


def _continuous_gripper(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite gripper value: {result}")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"Gripper value must be in [0, 1], got {result:g}")
    return result


def _quat_inverse(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def normalize_quat(quat: Sequence[float], *, min_norm: float, label: str) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite {label} quaternion: {values}")
    norm = float(np.linalg.norm(values))
    if norm < min_norm:
        raise ValueError(f"{label} quaternion norm is too small: {norm:.6f}")
    return values / norm


def _as_action8(values: Sequence[float], *, label: str) -> np.ndarray:
    action = np.asarray(values, dtype=np.float64).reshape(8).copy()
    if not np.all(np.isfinite(action)):
        raise ValueError(f"Non-finite {label}: {action}")
    return action


def _as_xyz(values: Sequence[float], *, label: str) -> np.ndarray:
    xyz = np.asarray(values, dtype=np.float64).reshape(3).copy()
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Non-finite {label}: {xyz}")
    return xyz


def _as_pose7(values: Sequence[float], *, label: str) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64).reshape(7).copy()
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"Non-finite {label}: {pose}")
    return pose


def _validate_xyz(xyz: Sequence[float], config: EefActionTransformConfig) -> None:
    values = _as_xyz(xyz, label="xyz")
    lower = np.asarray(config.xyz_min, dtype=np.float64)
    upper = np.asarray(config.xyz_max, dtype=np.float64)
    axes = [
        axis
        for axis, value, lo_i, hi_i in zip(
            ("x", "y", "z"), values, lower, upper, strict=True
        )
        if value < lo_i or value > hi_i
    ]
    if axes:
        raise ValueError(
            "EEF policy action is outside the configured workspace on "
            f"{','.join(axes)}: xyz={values.tolist()} "
            f"min={lower.tolist()} max={upper.tolist()}"
        )
