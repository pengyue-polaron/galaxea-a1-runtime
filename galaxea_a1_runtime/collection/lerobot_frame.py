"""Pure assembly and validation of one canonical A1 LeRobot frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from galaxea_a1_runtime.schema import (
    ACTION_FEATURE_KEY,
    A1_STATE_NAMES,
    FRONT_DEPTH_FEATURE_KEY,
    FRONT_IMAGE_FEATURE_KEY,
    JOINT_ACTION_NAMES_RAD,
    STATE_FEATURE_KEY,
    TASK_FEATURE_KEY,
    WRIST_IMAGE_FEATURE_KEY,
)


def build_lerobot_frame(
    *,
    state: tuple[float, ...],
    action: tuple[float, ...],
    front_bgr: Any,
    wrist_bgr: Any,
    task: str,
    front_depth_mm: Any | None = None,
) -> dict[str, Any]:
    """Convert one synchronized sample into the canonical LeRobot feature map."""

    state_array = _finite_vector(
        state, expected=len(A1_STATE_NAMES), label=STATE_FEATURE_KEY
    )
    action_array = _finite_vector(
        action, expected=len(JOINT_ACTION_NAMES_RAD), label=ACTION_FEATURE_KEY
    )
    _normalized_gripper(state_array[-1], label=STATE_FEATURE_KEY)
    _normalized_gripper(action_array[-1], label=ACTION_FEATURE_KEY)
    frame = {
        STATE_FEATURE_KEY: state_array,
        ACTION_FEATURE_KEY: action_array,
        FRONT_IMAGE_FEATURE_KEY: _bgr_to_rgb(front_bgr, label="front"),
        WRIST_IMAGE_FEATURE_KEY: _bgr_to_rgb(wrist_bgr, label="wrist"),
        TASK_FEATURE_KEY: task,
    }
    if front_depth_mm is not None:
        depth = np.asarray(front_depth_mm)
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(
                f"front depth must be uint16 HxW millimeters, got {depth.dtype} {depth.shape}"
            )
        frame[FRONT_DEPTH_FEATURE_KEY] = depth[..., None]
    return frame


def _finite_vector(values: Any, *, expected: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (expected,):
        raise ValueError(f"{label} must have shape ({expected},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    return result


def _normalized_gripper(value: float, *, label: str) -> None:
    if value < -1e-6 or value > 1.0 + 1e-6:
        raise ValueError(f"{label} gripper is outside normalized [0, 1]")


def _bgr_to_rgb(image: Any, *, label: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(
            f"{label} RGB frame must be uint8 HxWx3, got {value.dtype} {value.shape}"
        )
    return value[..., ::-1].copy()
