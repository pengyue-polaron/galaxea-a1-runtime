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
    """Normalize observed/initial EE state without workspace-clamping feedback."""

    action = _as_action8(action8, label="EE state condition")
    action[3:7] = normalize_quat(
        action[3:7], min_norm=config.min_quat_norm, label="EE state condition"
    )
    action[7] = _continuous_gripper(action[7])
    return action


def sanitize_policy_action(
    raw8: Sequence[float],
    config: EefActionTransformConfig,
) -> np.ndarray:
    """Normalize a model action and apply explicit output-side bounds."""

    action = _as_action8(raw8, label="EEF policy action")
    action[:3] = _clamp_xyz(action[:3], config)
    action[3:7] = normalize_quat(
        action[3:7], min_norm=config.min_quat_norm, label="EEF policy action"
    )
    action[7] = _command_gripper(action[7], config)
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
    origin8: Sequence[float],
    *,
    min_quat_norm: float,
) -> np.ndarray:
    """Compose an episode-relative xyz+xyzw action onto the episode origin."""

    relative = _as_action8(relative8, label="relative EEF policy action")
    origin = _as_action8(origin8, label="episode origin")
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
    origin8: Sequence[float],
    *,
    min_quat_norm: float,
) -> np.ndarray:
    """Express an absolute xyz+xyzw action relative to the episode origin."""

    absolute = _as_action8(absolute8, label="absolute A1 action")
    origin = _as_action8(origin8, label="episode origin")
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


def _command_gripper(value: float, config: EefActionTransformConfig) -> float:
    del config
    return _continuous_gripper(value)


def _continuous_gripper(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite gripper value: {result}")
    return float(np.clip(result, 0.0, 1.0))


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


def clamp_notes(
    raw8: Sequence[float],
    config: EefActionTransformConfig,
) -> list[str]:
    raw = _as_action8(raw8, label="EEF policy action")
    notes: list[str] = []
    xyz_min = np.asarray(config.xyz_min, dtype=np.float64)
    xyz_max = np.asarray(config.xyz_max, dtype=np.float64)
    axes = [
        axis
        for axis, value, lo_i, hi_i in zip(("x", "y", "z"), raw[:3], xyz_min, xyz_max)
        if value < lo_i or value > hi_i
    ]
    if axes:
        notes.append("workspace:" + ",".join(axes))
    if raw[7] < 0.0 or raw[7] > 1.0:
        notes.append("gripper:0..1")
    return notes


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


def _clamp_xyz(xyz: Sequence[float], config: EefActionTransformConfig) -> np.ndarray:
    values = _as_xyz(xyz, label="xyz")
    return np.minimum(
        np.maximum(values, np.asarray(config.xyz_min, dtype=np.float64)),
        np.asarray(config.xyz_max, dtype=np.float64),
    )
