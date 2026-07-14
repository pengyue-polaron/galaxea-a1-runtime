"""Command-line contract for the LingBot end-effector bridge."""

from __future__ import annotations

import argparse

from galaxea_a1_runtime.apps.lingbot.bridge import (
    A1LingBotEEBridge,
)
from galaxea_a1_runtime.hardware.web_preview import add_web_preview_arguments


def parse_args():
    p = argparse.ArgumentParser(description="LingBot-VA EE-pose bridge for Galaxea A1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1106)
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually publish EE commands. Default is dry-run.",
    )
    p.add_argument(
        "--step-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for Enter before each model inference chunk.",
    )
    p.add_argument(
        "--step-actions",
        action="store_true",
        default=False,
        help="Wait for Enter before every individual EE pose command inside the predicted chunk.",
    )
    p.add_argument(
        "--no-kv-update",
        action="store_true",
        default=False,
        help="Skip LingBot KV-cache update after executing a chunk; useful for isolated manual probing.",
    )
    p.add_argument(
        "--max-model-calls",
        type=int,
        default=1,
        help="Stop after this many model calls. 0 means run until q/Ctrl-C.",
    )
    p.add_argument(
        "--execute-frames",
        type=int,
        default=1,
        help="How many LingBot frame chunks to execute per model call",
    )
    p.add_argument(
        "--condition-on-ee-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include current/initial 8D EE pose as LingBot action-state conditioning.",
    )
    p.add_argument(
        "--initial-ee-pose",
        type=float,
        nargs=8,
        default=None,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW", "GRIP"),
        help="Fallback 8D EE state condition when live feedback is unavailable; gripper is normalized 0..1.",
    )
    p.add_argument(
        "--lingbot-frame-chunk-size",
        type=int,
        default=4,
        help="LingBot action-state frame dimension for first-frame conditioning.",
    )
    p.add_argument(
        "--lingbot-action-per-frame",
        type=int,
        default=20,
        help="LingBot action-state horizon dimension for first-frame conditioning.",
    )
    p.add_argument("--exec-rate", type=float, default=30.0)
    p.add_argument(
        "--print-actions", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--review-deadband",
        type=float,
        default=0.001,
        help="XYZ delta below this many meters is printed as hold for direction review.",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--max-camera-age", type=float, default=0.5)
    p.add_argument("--cam0-serial", default="341522300456")
    p.add_argument(
        "--cam0-auto-exposure", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--cam0-exposure", type=int, default=140)
    p.add_argument("--cam0-gain", type=int, default=32)
    p.add_argument(
        "--cam0-auto-white-balance", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--cam0-white-balance", type=int, default=4600)
    p.add_argument(
        "--cam0-crop-enabled", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--cam0-crop-x", type=int, default=0)
    p.add_argument("--cam0-crop-y", type=int, default=0)
    p.add_argument("--cam0-crop-width", type=int, default=640)
    p.add_argument("--cam0-crop-height", type=int, default=480)
    p.add_argument("--cam0-observation-key", default="observation.images.front")
    p.add_argument("--cam1-device", default="/dev/video0")
    p.add_argument("--cam1-backend", choices=("realsense", "v4l2"), default="v4l2")
    p.add_argument("--cam1-serial", default="")
    p.add_argument("--cam1-backend-api", default="v4l2")
    p.add_argument("--cam1-observation-key", default="observation.images.wrist")
    p.add_argument("--state-pose-topic", default="/end_effector_pose")
    p.add_argument("--state-gripper-topic", default="/gripper_stroke_host")
    p.add_argument("--cmd-pose-topic", default="/a1_ee_target")
    p.add_argument("--cmd-gripper-topic", required=True)
    p.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    p.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    p.add_argument("--relay-enable-timeout", type=float, default=2.0)
    p.add_argument(
        "--max-relay-status-age",
        type=float,
        default=1.0,
        help="Maximum age in seconds for trusting /a1_arm_relay_status while executing.",
    )
    p.add_argument("--command-frame", default="world")
    p.add_argument(
        "--action-pose-mode",
        choices=["absolute", "episode-relative"],
        default="absolute",
        help="Interpret model EEF poses as world-absolute or relative to the startup EEF pose.",
    )
    p.add_argument(
        "--orientation-mode",
        choices=["hold-current", "model-quat"],
        default="hold-current",
        help="Safest default holds current EE orientation; model-quat uses LingBot channels 3..6 directly.",
    )
    p.add_argument(
        "--eef-servo-gain",
        type=float,
        default=1.0,
        help="Gain >1 sends an amplified tracker target toward the policy target to compensate under-tracking.",
    )
    p.add_argument(
        "--eef-servo-max-extra",
        type=float,
        default=0.04,
        help="Maximum extra overshoot distance in meters when eef-servo-gain > 1. 0 means unlimited before workspace clamp.",
    )
    p.add_argument(
        "--eef-servo-settle",
        type=float,
        default=0.0,
        help="Seconds to hold each command and measure target tracking error after publish.",
    )
    p.add_argument(
        "--eef-servo-tolerance",
        type=float,
        default=0.01,
        help="XYZ norm tolerance in meters for servo settle/correction.",
    )
    p.add_argument(
        "--eef-servo-corrections",
        type=int,
        default=0,
        help="Additional correction publishes after settle if actual EEF is still far from the policy target.",
    )
    p.add_argument(
        "--cache-actual-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache measured EEF feedback instead of the tracker command; enable only for matching training data.",
    )
    p.add_argument("--xyz-min", type=float, nargs=3, default=[0.06, -0.27, 0.06])
    p.add_argument("--xyz-max", type=float, nargs=3, default=[0.44, 0.14, 0.50])
    p.add_argument("--min-quat-norm", type=float, default=0.25)
    p.add_argument("--max-feedback-age", type=float, default=0.5)
    p.add_argument("--feedback-wait-timeout", type=float, default=5.0)
    p.add_argument("--gripper-stroke-min", type=float, required=True)
    p.add_argument("--gripper-stroke-max", type=float, required=True)
    add_web_preview_arguments(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gripper_stroke_max <= args.gripper_stroke_min:
        raise ValueError(
            "--gripper-stroke-max must be greater than --gripper-stroke-min"
        )
    if not args.execute:
        print(
            "[Bridge] DRY RUN: not publishing robot commands. Pass --execute to move the robot."
        )
    bridge = A1LingBotEEBridge(args)
    try:
        bridge.run()
        return 0
    finally:
        bridge.close()
