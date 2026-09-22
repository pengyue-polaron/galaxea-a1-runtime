"""Pure intermediate-pose construction for bounded IK recovery."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IkSubgoalConfig:
    max_attempts: int
    max_steps: int
    max_translation_m: float
    max_rotation_rad: float
    max_joint_delta_rad: float
    feedback_timeout_s: float
    arrival_position_tolerance_m: float
    arrival_orientation_tolerance_rad: float
    min_translation_progress_m: float
    min_rotation_progress_rad: float


class IkSubgoalExecuted(Exception):
    """Bounded recovery reached the goal; its partial cache must be discarded."""


def pose_distance(
    xyz: Sequence[float], quat: Sequence[float], target: Sequence[float]
) -> tuple[float, float]:
    position, rotation = _pose(np.concatenate([xyz, quat]))
    target_position, target_rotation = _pose(target)
    dot = float(np.clip(abs(np.dot(rotation, target_rotation)), 0.0, 1.0))
    return float(np.linalg.norm(position - target_position)), 2 * math.acos(dot)


def intermediate_targets(
    xyz: Sequence[float],
    quat: Sequence[float],
    target: Sequence[float],
    config: IkSubgoalConfig,
) -> Iterator[tuple[float, np.ndarray]]:
    """Halve a short translation/shortest-arc rotation toward the requested pose."""
    current_xyz, current_quat = _pose(np.concatenate([xyz, quat]))
    target_xyz, target_quat = _pose(target)
    translation, rotation = pose_distance(current_xyz, current_quat, target)
    fraction = 1.0
    if translation > 0:
        fraction = min(fraction, config.max_translation_m / translation)
    if rotation > 0:
        fraction = min(fraction, config.max_rotation_rad / rotation)
    dot = float(np.dot(current_quat, target_quat))
    if dot < 0:
        target_quat = -target_quat
        dot = -dot
    angle = math.acos(float(np.clip(dot, 0.0, 1.0)))
    for _ in range(config.max_attempts):
        if angle < 1e-8:
            intermediate_quat = (1 - fraction) * current_quat + fraction * target_quat
        else:
            intermediate_quat = (
                math.sin((1 - fraction) * angle) * current_quat
                + math.sin(fraction * angle) * target_quat
            ) / math.sin(angle)
        result = np.asarray(target, dtype=np.float64).copy()
        result[:3] = current_xyz + fraction * (target_xyz - current_xyz)
        result[3:7] = intermediate_quat / np.linalg.norm(intermediate_quat)
        yield fraction, result
        fraction *= 0.5


def subgoal_has_progress(
    xyz: Sequence[float],
    quat: Sequence[float],
    start: Sequence[float],
    requested: Sequence[float],
    config: IkSubgoalConfig,
) -> bool:
    """Require measured movement and lower goal error independently of IK precision."""
    moved = pose_distance(xyz, quat, start)
    if (
        moved[0] <= config.min_translation_progress_m
        and moved[1] <= config.min_rotation_progress_rad
    ):
        return False
    start_xyz, start_quat = _pose(start)
    initial = pose_distance(start_xyz, start_quat, requested)
    remaining = pose_distance(xyz, quat, requested)

    def score(errors: tuple[float, float]) -> float:
        return (errors[0] / config.min_translation_progress_m) ** 2 + (
            errors[1] / config.min_rotation_progress_rad
        ) ** 2

    return score(remaining) < score(initial)


def _pose(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape not in {(7,), (8,)} or not np.all(np.isfinite(pose)):
        raise ValueError("Subgoal pose must contain seven or eight finite values")
    norm = float(np.linalg.norm(pose[3:7]))
    if norm <= 0:
        raise ValueError("Subgoal quaternion must have nonzero norm")
    return pose[:3], pose[3:7] / norm
