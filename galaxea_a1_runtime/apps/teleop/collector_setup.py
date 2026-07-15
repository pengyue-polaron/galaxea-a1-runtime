"""Pure validation and ROI construction for teleop collection."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi


def validate_collector_args(args: Any) -> tuple[StateMode, ImageRoi | None]:
    positive = {
        "--fps": args.fps,
        "--max-camera-age-s": args.max_camera_age_s,
        "--max-joint-feedback-age-s": args.max_joint_feedback_age_s,
        "--max-eef-feedback-age-s": args.max_eef_feedback_age_s,
        "--max-action-age-s": args.max_action_age_s,
        "--max-gripper-age-s": args.max_gripper_age_s,
        "--max-joint-action-step-rad": args.max_joint_action_step_rad,
    }
    for label, value in positive.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if args.gripper_stroke_max <= args.gripper_stroke_min:
        raise ValueError(
            "--gripper-stroke-max must be greater than --gripper-stroke-min"
        )
    if args.teleop_config is None:
        raise ValueError(
            "collection requires --teleop-config for reproducible metadata"
        )
    if args.auto_reset_after_save and args.reset_runtime_script is None:
        raise ValueError(
            "automatic reset requires --reset-runtime-script and --teleop-config"
        )
    if args.cam0_depth_enabled and (
        args.cam0_depth_width <= 0 or args.cam0_depth_height <= 0
    ):
        raise ValueError(
            "--cam0-depth-width and --cam0-depth-height must be positive when depth is enabled"
        )
    return StateMode(args.state_mode), _front_roi(args)


def _front_roi(args: Any) -> ImageRoi | None:
    if not args.cam0_crop_enabled:
        return None
    roi = ImageRoi(
        x=args.cam0_crop_x,
        y=args.cam0_crop_y,
        width=args.cam0_crop_width,
        height=args.cam0_crop_height,
    )
    roi.validate(
        image_width=args.cam0_width,
        image_height=args.cam0_height,
        label="AgentView collection ROI",
    )
    if roi.width != roi.height:
        raise ValueError("AgentView collection ROI must be square")
    if args.cam0_depth_enabled and not args.cam0_align_depth_to_color:
        raise ValueError(
            "AgentView depth must be aligned when collection crop is enabled"
        )
    return roi
