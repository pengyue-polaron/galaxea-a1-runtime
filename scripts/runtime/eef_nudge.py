#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive safe EEF nudge tool for hardware acceptance tests."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR)

import rospy
from geometry_msgs.msg import PoseStamped
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.eef_bridge import (
    EefCommandPublisher,
    pose_msg_to_xyz_quat,
)
from galaxea_a1_runtime.console import ArgumentParser, info, step, success
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache
from galaxea_a1_runtime.configuration.system import (
    DEFAULT_SYSTEM_CONFIG,
    SystemConfig,
    load_system_config,
)
from galaxea_a1_runtime.runtime.relay import (
    RelayStatus,
    decode_relay_status,
    relay_state_summary,
    relay_status_is_fresh,
)


class EefNudge:
    def __init__(
        self,
        system: SystemConfig,
        *,
        execute: bool,
        step_m: float,
        settle_s: float,
    ):
        self.system = system
        self.execute = execute
        self.step_m = step_m
        self.settle_s = settle_s
        self.pose = LatestMessageCache()
        self.relay = LatestMessageCache()
        topics = system.topics
        rospy.init_node("a1_eef_nudge", anonymous=False, disable_signals=True)
        pose_pub = rospy.Publisher(topics.eef_target, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(
            topics.gripper_target, gripper_position_control, queue_size=10
        )
        enable_pub = rospy.Publisher(
            topics.motion_enable, Bool, queue_size=1, latch=True
        )
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=pose_pub,
            gripper_pub=gripper_pub,
            motion_enable_pub=enable_pub,
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=system.eef.command_frame,
            gripper_to_stroke=lambda value: value,
            execute=execute,
        )
        rospy.Subscriber(topics.eef_pose, PoseStamped, self.pose.callback, queue_size=1)
        rospy.Subscriber(topics.relay_status, String, self._relay_cb, queue_size=1)
        self.keepalive = rospy.Timer(rospy.Duration(1.0 / 25.0), self._publish_hold)

    def _relay_cb(self, msg: String) -> None:
        self.relay.set(decode_relay_status(msg.data))

    def _publish_hold(self, _event=None) -> None:
        self.commander.publish_active_pose_target()

    def wait_pose(self) -> PoseStamped:
        deadline = time.monotonic() + self.system.eef.feedback_wait_timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg, _ = self.pose.snapshot()
            if pose_msg_to_xyz_quat(msg) is not None:
                return msg
            time.sleep(0.05)
        raise RuntimeError(f"No valid {self.system.topics.eef_pose} feedback")

    def enable_motion(self) -> None:
        if not self.execute:
            info(
                "EEF nudge is DRY RUN: relay and robot command publishing stay disabled."
            )
            return
        self.commander.publish_motion_enable(True)
        deadline = time.monotonic() + self.system.relay.enable_timeout_s
        last = "no relay status"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status, updated = self.relay.snapshot()
            last = relay_state_summary(
                status,
                updated,
                max_age_s=self.system.relay.max_status_age_s,
            )
            relay = status or RelayStatus(state="UNKNOWN")
            if relay_status_is_fresh(
                updated, max_age_s=self.system.relay.max_status_age_s
            ):
                if relay.state == "ACTIVE":
                    success("EEF nudge relay is ACTIVE.")
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
        info(f"Current EEF xyz={np.round(xyz, 4).tolist()}")
        self.enable_motion()

        for label, delta in sequence(self.step_m):
            command = (
                input(f"[eef-nudge] Enter=nudge {label}, s=skip, q=quit: ")
                .strip()
                .lower()
            )
            if command in {"q", "quit", "exit"}:
                break
            if command in {"s", "skip"}:
                continue
            current = self.wait_pose()
            xyz, quat = pose_msg_to_xyz_quat(current)
            target_xyz = xyz + delta
            action8 = np.concatenate([target_xyz, quat, [0.0]])
            self.commander.publish_action(action8, publish_gripper=False)
            step(
                f"Published {label} target_xyz={np.round(target_xyz, 4).tolist()} "
                f"delta_cm={np.round(delta * 100.0, 2).tolist()}"
            )
            time.sleep(self.settle_s)
            actual = self.wait_pose()
            actual_xyz, _ = pose_msg_to_xyz_quat(actual)
            info(
                f"Observed xyz={np.round(actual_xyz, 4).tolist()} "
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


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Step-gated EEF nudge through the safe A1 runtime."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Enable relay and publish commands."
    )
    parser.add_argument("--config", type=Path, default=ROOT_DIR / DEFAULT_SYSTEM_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step_m <= 0:
        raise ValueError("--step-m must be positive")
    if args.settle_s < 0:
        raise ValueError("--settle-s must be non-negative")
    system = load_system_config(args.config, repo_root=ROOT_DIR)
    nudge = EefNudge(
        system,
        execute=args.execute,
        step_m=system.eef_test.step_m,
        settle_s=system.eef_test.settle_s,
    )
    try:
        nudge.run()
    finally:
        nudge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
