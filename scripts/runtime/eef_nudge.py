#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive safe EEF nudge tool for hardware acceptance tests."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import rospy
from geometry_msgs.msg import PoseStamped
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.eef_bridge import (
    EefCommandPublisher,
    RelayStatus,
    decode_relay_status,
    pose_msg_to_xyz_quat,
    relay_state_summary,
    relay_status_is_fresh,
)


class Latest:
    def __init__(self):
        self._lock = threading.Lock()
        self.value: Any | None = None
        self.updated: float | None = None

    def set(self, value: Any) -> None:
        with self._lock:
            self.value = value
            self.updated = time.monotonic()

    def get(self) -> tuple[Any | None, float | None]:
        with self._lock:
            return self.value, self.updated


class EefNudge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pose = Latest()
        self.relay = Latest()
        rospy.init_node("a1_eef_nudge", anonymous=False, disable_signals=True)
        pose_pub = rospy.Publisher(args.cmd_pose_topic, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(args.cmd_gripper_topic, gripper_position_control, queue_size=10)
        enable_pub = rospy.Publisher(args.motion_enable_topic, Bool, queue_size=1, latch=True)
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=pose_pub,
            gripper_pub=gripper_pub,
            motion_enable_pub=enable_pub,
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=args.command_frame,
            gripper_to_stroke=lambda value: value,
            execute=args.execute,
        )
        rospy.Subscriber(args.state_pose_topic, PoseStamped, self.pose.set, queue_size=1)
        rospy.Subscriber(args.relay_status_topic, String, self._relay_cb, queue_size=1)
        self.keepalive = rospy.Timer(rospy.Duration(1.0 / args.hold_hz), self._publish_hold)

    def _relay_cb(self, msg: String) -> None:
        self.relay.set(decode_relay_status(msg.data))

    def _publish_hold(self, _event=None) -> None:
        self.commander.publish_active_pose_target()

    def wait_pose(self) -> PoseStamped:
        deadline = time.monotonic() + self.args.feedback_timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg, _ = self.pose.get()
            if pose_msg_to_xyz_quat(msg) is not None:
                return msg
            time.sleep(0.05)
        raise RuntimeError(f"No valid {self.args.state_pose_topic} feedback")

    def enable_motion(self) -> None:
        if not self.args.execute:
            print("[eef-nudge] DRY RUN: not enabling relay or publishing robot commands.")
            return
        self.commander.publish_motion_enable(True)
        deadline = time.monotonic() + self.args.relay_enable_timeout_s
        last = "no relay status"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status, updated = self.relay.get()
            last = relay_state_summary(
                status,
                updated,
                max_age_s=self.args.max_relay_status_age_s,
            )
            relay = status or RelayStatus(state="UNKNOWN")
            if relay_status_is_fresh(updated, max_age_s=self.args.max_relay_status_age_s):
                if relay.state == "ACTIVE":
                    print("[eef-nudge] relay ACTIVE")
                    return
                if relay.state == "FAULT":
                    break
            time.sleep(0.05)
        self.commander.publish_motion_enable(False)
        raise RuntimeError(f"Relay did not become ACTIVE: {last}")

    def run(self) -> None:
        current = self.wait_pose()
        self.commander.set_active_pose_target_from_msg(current)
        self.commander.publish_active_pose_target()
        xyz, quat = pose_msg_to_xyz_quat(current)
        print(f"[eef-nudge] current_xyz={np.round(xyz, 4).tolist()}")
        self.enable_motion()

        for label, delta in sequence(self.args.step_m):
            command = input(f"[eef-nudge] Enter=nudge {label}, s=skip, q=quit: ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command in {"s", "skip"}:
                continue
            current = self.wait_pose()
            xyz, quat = pose_msg_to_xyz_quat(current)
            target_xyz = xyz + delta
            action8 = np.concatenate([target_xyz, quat, [0.0]])
            self.commander.publish_action(action8, publish_gripper=False)
            print(
                f"[eef-nudge] published {label} target_xyz={np.round(target_xyz, 4).tolist()} "
                f"delta_cm={np.round(delta * 100.0, 2).tolist()}"
            )
            time.sleep(self.args.settle_s)
            actual = self.wait_pose()
            actual_xyz, _ = pose_msg_to_xyz_quat(actual)
            print(
                f"[eef-nudge] actual_xyz={np.round(actual_xyz, 4).tolist()} "
                f"observed_delta_cm={np.round((actual_xyz - xyz) * 100.0, 2).tolist()}"
            )

    def close(self) -> None:
        self.commander.publish_motion_enable(False)
        self.keepalive.shutdown()


def sequence(step_m: float) -> tuple[tuple[str, np.ndarray], ...]:
    step = float(step_m)
    return (
        ("x+", np.array([step, 0.0, 0.0], dtype=np.float64)),
        ("x-", np.array([-step, 0.0, 0.0], dtype=np.float64)),
        ("y+", np.array([0.0, step, 0.0], dtype=np.float64)),
        ("y-", np.array([0.0, -step, 0.0], dtype=np.float64)),
        ("z+", np.array([0.0, 0.0, step], dtype=np.float64)),
        ("z-", np.array([0.0, 0.0, -step], dtype=np.float64)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-gated EEF nudge through the safe A1 runtime.")
    parser.add_argument("--execute", action="store_true", help="Enable relay and publish commands.")
    parser.add_argument("--step-m", type=float, default=0.03)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--hold-hz", type=float, default=25.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=5.0)
    parser.add_argument("--relay-enable-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-relay-status-age-s", type=float, default=1.0)
    parser.add_argument("--state-pose-topic", default="/end_effector_pose")
    parser.add_argument("--cmd-pose-topic", default="/a1_ee_target")
    parser.add_argument("--cmd-gripper-topic", default="/gripper_position_control_host")
    parser.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--command-frame", default="world")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step_m <= 0:
        raise ValueError("--step-m must be positive")
    if args.hold_hz <= 0:
        raise ValueError("--hold-hz must be positive")
    nudge = EefNudge(args)
    try:
        nudge.run()
    finally:
        nudge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
