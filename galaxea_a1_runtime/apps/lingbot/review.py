"""Operator-facing LingBot action preview formatting."""

from __future__ import annotations

import numpy as np

from galaxea_a1_runtime.apps.eef_bridge import format_xyz_direction
from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionTransformConfig,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.lingbot.episode_state import LingBotEpisodeState
from galaxea_a1_runtime.console import info, step


class LingBotActionReviewer:
    def __init__(
        self,
        *,
        state: LingBotEpisodeState,
        action_config: LingBotActionTransformConfig,
        review_deadband_m: float,
        servo_gain: float,
        orientation_mode: str,
        execute: bool,
    ) -> None:
        self.state = state
        self.action_config = action_config
        self.review_deadband_m = review_deadband_m
        self.servo_gain = servo_gain
        self.orientation_mode = orientation_mode
        self.execute = execute

    def print_step(
        self,
        *,
        call_index: int,
        frame_index: int,
        step_index: int,
        model_action: np.ndarray,
        safe_action: np.ndarray,
    ) -> None:
        current = self.state.current_xyz()
        if current is None:
            condition_action = self.state.current_absolute_action()
            if condition_action is not None:
                current = condition_action[:3]
        absolute = self.state.model_to_absolute(model_action)
        raw_xyz = np.asarray(absolute[:3], dtype=np.float64)
        safe_delta = None if current is None else safe_action[:3] - current
        raw_delta = None if current is None else raw_xyz - current
        gripper_mm = gripper_stroke_from_norm(float(safe_action[7]), self.action_config)
        notes = self.state.clamp_notes(model_action)
        step(
            f"LingBot action call={call_index + 1} frame={frame_index} "
            f"step={step_index} "
            f"model={np.round(model_action, 4).tolist()} "
            f"absolute={np.round(absolute, 4).tolist()} "
            f"safe={np.round(safe_action, 4).tolist()}"
        )
        if current is not None:
            info(
                "current_xyz="
                f"{np.round(current, 4).tolist()} "
                f"raw_delta_cm={np.round(raw_delta * 100.0, 2).tolist()} "
                f"safe_delta_cm={np.round(safe_delta * 100.0, 2).tolist()} "
                f"safe_norm_cm={np.linalg.norm(safe_delta) * 100.0:.2f} "
                "direction="
                f"{format_xyz_direction(safe_delta, deadband_m=self.review_deadband_m)}"
            )
            tracker_command = self.state.tracker_command(safe_action)
            if not np.allclose(tracker_command[:3], safe_action[:3], atol=1e-5):
                tracker_delta = tracker_command[:3] - current
                info(
                    "tracker_cmd_xyz="
                    f"{np.round(tracker_command[:3], 4).tolist()} "
                    f"tracker_cmd_delta_cm={np.round(tracker_delta * 100.0, 2).tolist()} "
                    f"servo_gain={self.servo_gain:.2f}"
                )
        info(
            f"gripper_norm={safe_action[7]:.3f} "
            f"gripper_mm={gripper_mm:.1f} "
            f"orientation_mode={self.orientation_mode} execute={self.execute} "
            f"clamp={','.join(notes) if notes else 'none'}"
        )
