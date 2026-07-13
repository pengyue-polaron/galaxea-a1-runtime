"""Pure LingBot-VA action transforms for the Galaxea A1 app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

OrientationMode = Literal["hold-current", "model-quat"]


@dataclass(frozen=True)
class LingBotActionConfig:
    xyz_min: tuple[float, float, float] = (0.06, -0.27, 0.06)
    xyz_max: tuple[float, float, float] = (0.44, 0.14, 0.50)
    min_quat_norm: float = 0.25
    orientation_mode: OrientationMode = "hold-current"
    eef_servo_gain: float = 1.0
    eef_servo_max_extra: float = 0.04
    gripper_stroke_scale: float = 200.0
    gripper_stroke_offset: float = 0.0
    gripper_stroke_min: float = 0.0
    gripper_stroke_max: float = 200.0
    gripper_command_open_threshold: float = 0.5
    gripper_feedback_open_threshold_mm: float = 30.0


def normalize_condition_action(action8: Sequence[float], config: LingBotActionConfig) -> np.ndarray:
    """Normalize observed/initial EE state without workspace-clamping feedback."""

    action = _as_action8(action8, label="EE state condition")
    action[3:7] = normalize_quat(action[3:7], min_norm=config.min_quat_norm, label="EE state condition")
    action[7] = _binary_gripper(action[7], config.gripper_command_open_threshold)
    return action


def sanitize_policy_action(
    raw8: Sequence[float],
    config: LingBotActionConfig,
    *,
    current_xyz: Sequence[float] | None,
) -> np.ndarray:
    """Normalize a model action and apply explicit output-side bounds."""

    action = _as_action8(raw8, label="LingBot action")
    del current_xyz
    action[:3] = _clamp_xyz(action[:3], config)
    action[3:7] = normalize_quat(action[3:7], min_norm=config.min_quat_norm, label="LingBot action")
    action[7] = _binary_gripper(action[7], config.gripper_command_open_threshold)
    return action


def apply_orientation_mode(
    action8: Sequence[float],
    config: LingBotActionConfig,
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
    action[3:7] = normalize_quat(current_quat, min_norm=config.min_quat_norm, label="current EE orientation")
    return action


def prepare_policy_action(
    raw8: Sequence[float],
    config: LingBotActionConfig,
    *,
    current_xyz: Sequence[float] | None,
    current_quat: Sequence[float] | None,
    require_current_orientation: bool,
) -> np.ndarray:
    action = sanitize_policy_action(raw8, config, current_xyz=current_xyz)
    return apply_orientation_mode(
        action,
        config,
        current_quat=current_quat,
        require_current=require_current_orientation,
    )


def tracker_command_action(
    policy_action8: Sequence[float],
    config: LingBotActionConfig,
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


def gripper_norm_from_stroke(stroke_mm: float, config: LingBotActionConfig) -> float:
    return 1.0 if float(stroke_mm) >= config.gripper_feedback_open_threshold_mm else 0.0


def gripper_stroke_from_norm(norm: float, config: LingBotActionConfig) -> float:
    return (
        config.gripper_stroke_max
        if float(norm) >= config.gripper_command_open_threshold
        else config.gripper_stroke_min
    )


def _binary_gripper(value: float, threshold: float) -> float:
    return 1.0 if float(value) >= threshold else 0.0


def clamp_notes(
    raw8: Sequence[float],
    config: LingBotActionConfig,
    *,
    current_xyz: Sequence[float] | None,
) -> list[str]:
    raw = _as_action8(raw8, label="LingBot action")
    del current_xyz
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


def _clamp_xyz(xyz: Sequence[float], config: LingBotActionConfig) -> np.ndarray:
    values = _as_xyz(xyz, label="xyz")
    return np.minimum(
        np.maximum(values, np.asarray(config.xyz_min, dtype=np.float64)),
        np.asarray(config.xyz_max, dtype=np.float64),
    )
