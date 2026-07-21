"""Fresh feedback and episode-coordinate state shared by EEF policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from galaxea_a1_runtime.hardware.eef_bridge import pose_msg_to_xyz_quat
from galaxea_a1_runtime.policies.eef_actions import (
    EefActionTransformConfig,
    absolute_action_to_relative,
    gripper_norm_from_stroke,
    normalize_condition_action,
    relative_action_to_absolute,
    validate_policy_action,
)
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


class EefPolicyState:
    """Own live EEF feedback and the model/world action coordinate transform."""

    def __init__(
        self,
        *,
        action_config: EefActionTransformConfig,
        pose_mode: str,
        max_feedback_age_s: float,
    ) -> None:
        self.action_config = action_config
        self.pose_mode = pose_mode
        self.max_feedback_age_s = max_feedback_age_s
        self.pose_feedback: LatestMessageCache[Any] = LatestMessageCache()
        self.gripper_feedback: LatestMessageCache[float] = LatestMessageCache()
        self.episode_origin: np.ndarray | None = None

    def pose_callback(self, msg: Any) -> None:
        if (
            pose_msg_to_xyz_quat(msg, min_quat_norm=self.action_config.min_quat_norm)
            is None
        ):
            self.pose_feedback.clear()
            return
        self.pose_feedback.set(msg)

    def gripper_callback(self, msg: Any) -> None:
        positions = getattr(msg, "position", ())
        if not positions:
            self.gripper_feedback.clear()
            return
        try:
            value = float(positions[0])
            gripper_norm_from_stroke(value, self.action_config)
        except (OverflowError, TypeError, ValueError):
            self.gripper_feedback.clear()
            return
        self.gripper_feedback.set(value)

    def pose_is_fresh(self) -> bool:
        return self.current_xyz_quat() is not None

    def gripper_is_fresh(self) -> bool:
        return self.gripper_feedback.get(max_age_s=self.max_feedback_age_s) is not None

    def current_pose_message(self) -> Any | None:
        return self.pose_feedback.get(max_age_s=self.max_feedback_age_s)

    def current_xyz_quat(self) -> tuple[np.ndarray, np.ndarray] | None:
        return pose_msg_to_xyz_quat(
            self.current_pose_message(),
            min_quat_norm=self.action_config.min_quat_norm,
        )

    def current_xyz(self) -> np.ndarray | None:
        value = self.current_xyz_quat()
        return None if value is None else value[0]

    def current_absolute_action(self) -> np.ndarray | None:
        pose = self.current_xyz_quat()
        gripper = self.gripper_feedback.get(max_age_s=self.max_feedback_age_s)
        if pose is None or gripper is None:
            return None
        xyz, quat = pose
        gripper_norm = gripper_norm_from_stroke(gripper, self.action_config)
        return self._normalize(np.concatenate([xyz, quat, [gripper_norm]]))

    def ensure_episode_origin(self) -> np.ndarray | None:
        if self.episode_origin is None:
            pose = self.current_xyz_quat()
            if pose is not None:
                self.episode_origin = np.concatenate(pose)
        return self.episode_origin

    def absolute_to_model(self, absolute8: Sequence[float]) -> np.ndarray:
        absolute = np.asarray(absolute8, dtype=np.float64).reshape(8)
        if self.pose_mode == "absolute":
            return absolute.copy()
        return absolute_action_to_relative(
            absolute,
            self._require_origin(),
            min_quat_norm=self.action_config.min_quat_norm,
        )

    def model_to_absolute(self, model8: Sequence[float]) -> np.ndarray:
        model = np.asarray(model8, dtype=np.float64).reshape(8)
        if self.pose_mode == "absolute":
            return model.copy()
        return relative_action_to_absolute(
            model,
            self._require_origin(),
            min_quat_norm=self.action_config.min_quat_norm,
        )

    def validate(self, model8: Sequence[float]) -> np.ndarray:
        return validate_policy_action(
            self.model_to_absolute(model8),
            self.action_config,
        )

    def _normalize(self, action8: Sequence[float]) -> np.ndarray:
        return normalize_condition_action(action8, self.action_config)

    def _require_origin(self) -> np.ndarray:
        origin = self.ensure_episode_origin()
        if origin is None:
            raise RuntimeError("Episode EEF origin is unavailable")
        return origin
