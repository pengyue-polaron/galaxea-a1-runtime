"""Operator-facing EEF policy action preview formatting."""

from __future__ import annotations

import numpy as np

from galaxea_a1_runtime.apps.eef_bridge import format_xyz_direction
from galaxea_a1_runtime.apps.eef_policy_actions import (
    EefActionTransformConfig,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.eef_policy_state import EefPolicyState
from galaxea_a1_runtime.console import info, step


class EefActionReviewer:
    def __init__(
        self,
        *,
        state: EefPolicyState,
        action_config: EefActionTransformConfig,
        review_deadband_m: float,
        execute: bool,
        policy_label: str,
    ) -> None:
        self.state = state
        self.action_config = action_config
        self.review_deadband_m = review_deadband_m
        self.execute = execute
        self.policy_label = policy_label

    def print_step(
        self,
        *,
        call_index: int,
        frame_index: int,
        step_index: int,
        model_action: np.ndarray,
        validated_action: np.ndarray,
    ) -> None:
        current = self.state.current_xyz()
        if current is None:
            condition_action = self.state.current_absolute_action()
            if condition_action is not None:
                current = condition_action[:3]
        absolute = self.state.model_to_absolute(model_action)
        raw_xyz = np.asarray(absolute[:3], dtype=np.float64)
        validated_delta = None if current is None else validated_action[:3] - current
        raw_delta = None if current is None else raw_xyz - current
        gripper_mm = gripper_stroke_from_norm(
            float(validated_action[7]), self.action_config
        )
        step(
            f"{self.policy_label} action call={call_index + 1} frame={frame_index} "
            f"step={step_index} "
            f"model={np.round(model_action, 4).tolist()} "
            f"absolute={np.round(absolute, 4).tolist()} "
            f"validated={np.round(validated_action, 4).tolist()}"
        )
        if current is not None:
            info(
                "current_xyz="
                f"{np.round(current, 4).tolist()} "
                f"raw_delta_cm={np.round(raw_delta * 100.0, 2).tolist()} "
                f"validated_delta_cm={np.round(validated_delta * 100.0, 2).tolist()} "
                f"validated_norm_cm={np.linalg.norm(validated_delta) * 100.0:.2f} "
                "direction="
                f"{format_xyz_direction(validated_delta, deadband_m=self.review_deadband_m)}"
            )
        info(
            f"gripper_norm={validated_action[7]:.3f} "
            f"gripper_mm={gripper_mm:.1f} "
            f"execute={self.execute}"
        )
