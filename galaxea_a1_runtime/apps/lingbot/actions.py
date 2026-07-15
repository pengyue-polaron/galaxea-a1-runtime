"""Pure LingBot-VA action transforms for the Galaxea A1 app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.gripper import denormalize_stroke, normalize_stroke

OrientationMode = Literal["hold-current", "model-quat"]


@dataclass(frozen=True, kw_only=True)
class LingBotActionTransformConfig:
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    min_quat_norm: float
    orientation_mode: OrientationMode
    gripper_stroke_min: float
    gripper_stroke_max: float
    eef_servo_gain: float
    eef_servo_max_extra: float


def build_action_transform_config(
    config: LingBotConfig,
) -> LingBotActionTransformConfig:
    """Derive the pure action transform from its deployment and system owners."""

    system = config.system
    return LingBotActionTransformConfig(
        xyz_min=system.eef.xyz_min,
        xyz_max=system.eef.xyz_max,
        min_quat_norm=system.eef.min_quat_norm,
        orientation_mode=system.eef.orientation_mode,
        gripper_stroke_min=system.gripper.stroke_min_mm,
        gripper_stroke_max=system.gripper.stroke_max_mm,
        eef_servo_gain=config.servo.gain,
        eef_servo_max_extra=config.servo.max_extra_m,
    )


def normalize_condition_action(
    action8: Sequence[float], config: LingBotActionTransformConfig
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
    config: LingBotActionTransformConfig,
) -> np.ndarray:
    """Normalize a model action and apply explicit output-side bounds."""

    action = _as_action8(raw8, label="LingBot action")
    action[:3] = _clamp_xyz(action[:3], config)
    action[3:7] = normalize_quat(
        action[3:7], min_norm=config.min_quat_norm, label="LingBot action"
    )
    action[7] = _command_gripper(action[7], config)
    return action


def apply_orientation_mode(
    action8: Sequence[float],
    config: LingBotActionTransformConfig,
    *,
    current_quat: Sequence[float] | None,
    require_current: bool,
) -> np.ndarray:
    """Apply the app's explicit orientation policy."""

    action = _as_action8(action8, label="LingBot action")
    if config.orientation_mode == "model-quat":
        return action
    if config.orientation_mode != "hold-current":
        raise ValueError(f"unsupported orientation mode: {config.orientation_mode}")
    if current_quat is None:
        if require_current:
            raise RuntimeError("No valid current EE orientation; refusing to publish")
        return action
    action[3:7] = normalize_quat(
        current_quat, min_norm=config.min_quat_norm, label="current EE orientation"
    )
    return action


def prepare_policy_action(
    raw8: Sequence[float],
    config: LingBotActionTransformConfig,
    *,
    current_quat: Sequence[float] | None,
    require_current_orientation: bool,
) -> np.ndarray:
    action = sanitize_policy_action(raw8, config)
    return apply_orientation_mode(
        action,
        config,
        current_quat=current_quat,
        require_current=require_current_orientation,
    )


def tracker_command_action(
    policy_action8: Sequence[float],
    config: LingBotActionTransformConfig,
    *,
    current_xyz: Sequence[float] | None,
) -> np.ndarray:
    """Optionally amplify the tracker target when servo compensation is enabled."""

    command = _as_action8(policy_action8, label="policy action")
    if current_xyz is None or config.eef_servo_gain <= 1.0:
        return command

    cur = _as_xyz(current_xyz, label="current_xyz")
    residual = command[:3] - cur
    extra = (config.eef_servo_gain - 1.0) * residual
    if config.eef_servo_max_extra > 0:
        extra_norm = float(np.linalg.norm(extra))
        if extra_norm > config.eef_servo_max_extra:
            extra *= config.eef_servo_max_extra / extra_norm
    command[:3] = _clamp_xyz(command[:3] + extra, config)
    return command


def gripper_norm_from_stroke(
    stroke_mm: float, config: LingBotActionTransformConfig
) -> float:
    return normalize_stroke(
        stroke_mm,
        stroke_min_mm=config.gripper_stroke_min,
        stroke_max_mm=config.gripper_stroke_max,
    )


def gripper_stroke_from_norm(
    norm: float, config: LingBotActionTransformConfig
) -> float:
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

    relative = _as_action8(relative8, label="relative LingBot action")
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


def _command_gripper(value: float, config: LingBotActionTransformConfig) -> float:
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
    config: LingBotActionTransformConfig,
) -> list[str]:
    raw = _as_action8(raw8, label="LingBot action")
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


def _clamp_xyz(
    xyz: Sequence[float], config: LingBotActionTransformConfig
) -> np.ndarray:
    values = _as_xyz(xyz, label="xyz")
    return np.minimum(
        np.maximum(values, np.asarray(config.xyz_min, dtype=np.float64)),
        np.asarray(config.xyz_max, dtype=np.float64),
    )
