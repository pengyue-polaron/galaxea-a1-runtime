"""Command-line contract for the ACT joint bridge."""

from __future__ import annotations

import argparse

import numpy as np

from galaxea_a1_runtime.apps.act.bridge import ActJointBridge, _front_roi_from_args
from galaxea_a1_runtime.hardware.web_preview import add_web_preview_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-backbone-download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--execute", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--step-mode", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--execute-steps-per-inference", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-model-calls", type=int, default=0)
    parser.add_argument(
        "--print-actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preview-steps", type=int, default=5)
    parser.add_argument("--joint-states-topic", default="/joint_states_host")
    parser.add_argument("--target-topic", default="/arm_joint_target_position")
    parser.add_argument(
        "--staged-command-topic", default="/arm_joint_command_a1_staged"
    )
    parser.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--gripper-target-topic", required=True)
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument("--relay-enable-timeout", type=float, default=2.0)
    parser.add_argument("--max-relay-status-age", type=float, default=1.0)
    parser.add_argument("--target-joint-names", nargs=6, required=True)
    parser.add_argument("--lower-limits", nargs=6, type=float, required=True)
    parser.add_argument("--upper-limits", nargs=6, type=float, required=True)
    parser.add_argument(
        "--action-step-guard-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-joint-action-step-rad", type=float, default=0.25)
    parser.add_argument("--max-first-target-delta-rad", type=float, default=0.25)
    parser.add_argument("--initial-alignment-tolerance", type=float, default=0.05)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--max-feedback-age", type=float, default=0.5)
    parser.add_argument("--max-camera-age", type=float, default=0.5)
    parser.add_argument("--gripper-stroke-min", type=float, required=True)
    parser.add_argument("--gripper-stroke-max", type=float, required=True)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=20)
    parser.add_argument("--cam0-serial", default="")
    parser.add_argument(
        "--cam0-auto-exposure", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cam0-exposure", type=int, default=140)
    parser.add_argument("--cam0-gain", type=int, default=32)
    parser.add_argument(
        "--cam0-auto-white-balance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cam0-white-balance", type=int, default=4600)
    parser.add_argument(
        "--cam0-crop-enabled", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cam0-crop-x", type=int, default=0)
    parser.add_argument("--cam0-crop-y", type=int, default=0)
    parser.add_argument("--cam0-crop-width", type=int, default=640)
    parser.add_argument("--cam0-crop-height", type=int, default=480)
    parser.add_argument("--cam1-device", default="auto")
    parser.add_argument("--cam1-backend", choices=("realsense", "v4l2"), default="v4l2")
    parser.add_argument("--cam1-serial", default="")
    parser.add_argument("--cam1-backend-api", default="v4l2")
    parser.add_argument("--cam1-pixel-format", default="YUYV")
    add_web_preview_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "execute_steps_per_inference",
        "preview_steps",
        "cam_width",
        "cam_height",
        "cam_fps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "control_hz",
        "relay_enable_timeout",
        "max_relay_status_age",
        "initial_alignment_tolerance",
        "state_timeout",
        "max_feedback_age",
        "max_camera_age",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.action_step_guard_enabled:
        for name in ("max_joint_action_step_rad", "max_first_target_delta_rad"):
            if float(getattr(args, name)) <= 0:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be positive when the guard is enabled"
                )
    if args.max_model_calls < 0:
        raise ValueError("--max-model-calls must be >= 0")
    if args.gripper_stroke_max <= args.gripper_stroke_min:
        raise ValueError(
            "--gripper-stroke-max must be greater than --gripper-stroke-min"
        )
    lower = np.asarray(args.lower_limits, dtype=np.float64)
    upper = np.asarray(args.upper_limits, dtype=np.float64)
    if np.any(lower >= upper):
        raise ValueError("--lower-limits must be below --upper-limits")
    _front_roi_from_args(args)


def main() -> int:
    args = parse_args()
    validate_args(args)
    bridge: ActJointBridge | None = None
    try:
        bridge = ActJointBridge(args)
        bridge.run()
        return 0
    finally:
        if bridge is not None:
            bridge.close()
