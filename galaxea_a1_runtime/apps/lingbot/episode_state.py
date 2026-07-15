"""Fresh feedback and episode-coordinate state for LingBot execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from galaxea_a1_runtime.apps.eef_bridge import (
    condition_state_from_action8,
    pose_msg_to_xyz_quat,
)
from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionTransformConfig,
    absolute_action_to_relative,
    clamp_notes,
    gripper_norm_from_stroke,
    normalize_condition_action,
    prepare_policy_action,
    relative_action_to_absolute,
    sanitize_policy_action,
    tracker_command_action,
)
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


class LingBotEpisodeState:
    """Own live EEF feedback and the model/world action coordinate transform."""

    def __init__(
        self,
        *,
        action_config: LingBotActionTransformConfig,
        pose_mode: str,
        max_feedback_age_s: float,
        initial_action8: Sequence[float] | None,
        frame_chunk_size: int,
        action_per_frame: int,
    ) -> None:
        self.action_config = action_config
        self.pose_mode = pose_mode
        self.max_feedback_age_s = max_feedback_age_s
        self.initial_action8 = (
            None
            if initial_action8 is None
            else np.asarray(initial_action8, dtype=np.float64).reshape(8).copy()
        )
        self.frame_chunk_size = frame_chunk_size
        self.action_per_frame = action_per_frame
        self.pose_feedback: LatestMessageCache[Any] = LatestMessageCache()
        self.gripper_feedback: LatestMessageCache[float] = LatestMessageCache()
        self.episode_origin: np.ndarray | None = None

    def pose_callback(self, msg: Any) -> None:
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
        return self.pose_feedback.get(max_age_s=self.max_feedback_age_s) is not None

    def gripper_is_fresh(self) -> bool:
        return self.gripper_feedback.get(max_age_s=self.max_feedback_age_s) is not None

    def current_pose_message(self) -> Any | None:
        return self.pose_feedback.get()

    def current_xyz_quat(self) -> tuple[np.ndarray, np.ndarray] | None:
        return pose_msg_to_xyz_quat(self.current_pose_message())

    def current_xyz(self) -> np.ndarray | None:
        value = self.current_xyz_quat()
        return None if value is None else value[0]

    def current_quat(self) -> np.ndarray | None:
        value = self.current_xyz_quat()
        return None if value is None else value[1]

    def current_absolute_action(self) -> np.ndarray | None:
        if not self.pose_is_fresh() or not self.gripper_is_fresh():
            return self._initial_action()
        pose = self.current_xyz_quat()
        gripper = self.gripper_feedback.get(max_age_s=self.max_feedback_age_s)
        if pose is None or gripper is None:
            return None
        xyz, quat = pose
        gripper_norm = gripper_norm_from_stroke(gripper, self.action_config)
        return self._normalize(np.concatenate([xyz, quat, [gripper_norm]]))

    def ensure_episode_origin(self) -> np.ndarray | None:
        if self.episode_origin is None:
            initial = self._initial_action()
            self.episode_origin = (
                initial if initial is not None else self.current_absolute_action()
            )
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

    def model_condition(self) -> np.ndarray | None:
        absolute = self.current_absolute_action()
        if absolute is None:
            return None
        model = self._normalize(self.absolute_to_model(absolute))
        return condition_state_from_action8(
            model,
            frame_chunk_size=self.frame_chunk_size,
            action_per_frame=self.action_per_frame,
        )

    def sanitize(self, model8: Sequence[float]) -> np.ndarray:
        return sanitize_policy_action(
            self.model_to_absolute(model8),
            self.action_config,
        )

    def prepare(
        self, model8: Sequence[float], *, require_orientation: bool
    ) -> np.ndarray:
        return prepare_policy_action(
            self.model_to_absolute(model8),
            self.action_config,
            current_quat=self.current_quat(),
            require_current_orientation=require_orientation,
        )

    def tracker_command(self, policy_action8: Sequence[float]) -> np.ndarray:
        return tracker_command_action(
            policy_action8,
            self.action_config,
            current_xyz=self.current_xyz(),
        )

    def measured_action(self, fallback: Sequence[float]) -> np.ndarray:
        actual = np.asarray(fallback, dtype=np.float64).reshape(8).copy()
        pose = self.current_xyz_quat()
        if pose is not None:
            actual[:3], actual[3:7] = pose
        gripper = self.gripper_feedback.get(max_age_s=self.max_feedback_age_s)
        if gripper is not None:
            actual[7] = gripper_norm_from_stroke(gripper, self.action_config)
        return actual

    def clamp_notes(self, model8: Sequence[float]) -> list[str]:
        return clamp_notes(
            self.model_to_absolute(model8),
            self.action_config,
        )

    def _initial_action(self) -> np.ndarray | None:
        if self.initial_action8 is None:
            return None
        return self._normalize(self.initial_action8)

    def _normalize(self, action8: Sequence[float]) -> np.ndarray:
        return normalize_condition_action(action8, self.action_config)

    def _require_origin(self) -> np.ndarray:
        origin = self.ensure_episode_origin()
        if origin is None:
            raise RuntimeError("Episode EEF origin is unavailable")
        return origin
