#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive EEF-to-IK nudge tool for staged hardware acceptance tests."""

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
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.eef_bridge import (
    EefIkCommandPublisher,
    pose_msg_to_xyz_quat,
)
from galaxea_a1_runtime.console import ArgumentParser, info, step, success, warning
from galaxea_a1_runtime.hardware.eef_ik import build_eef_ik_solver
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
from galaxea_a1_runtime.runtime.ros_feedback import (
    A1JointStateCache,
    StagedCommandMonitor,
    wait_for_staged_joint_alignment,
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
        self.joints = A1JointStateCache(system.joint_safety.names)
        self.staged = StagedCommandMonitor()
        self.relay = LatestMessageCache()
        topics = system.topics
        rospy.init_node("a1_eef_nudge", anonymous=False, disable_signals=True)
        gripper_pub = rospy.Publisher(
            topics.gripper_target, gripper_position_control, queue_size=10
        )
        enable_pub = rospy.Publisher(
            topics.motion_enable, Bool, queue_size=1, latch=True
        )
        self.commander = EefIkCommandPublisher(
            rospy=rospy,
            target_pub=rospy.Publisher(topics.joint_target, JointState, queue_size=10),
            gripper_pub=gripper_pub,
            motion_enable_pub=enable_pub,
            joint_state_msg_type=JointState,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            joint_names=system.joint_safety.names,
            current_joint_positions=lambda: self.joints.positions(
                max_age_s=system.joint_safety.max_feedback_age_s
            ),
            solver=build_eef_ik_solver(system),
            gripper_to_stroke=lambda value: value,
            execute=execute,
        )
        rospy.Subscriber(topics.eef_pose, PoseStamped, self.pose.callback, queue_size=1)
        rospy.Subscriber(
            topics.joint_states, JointState, self.joints.callback, queue_size=1
        )
        rospy.Subscriber(
            topics.staged_command, arm_control, self.staged.callback, queue_size=1
        )
        rospy.Subscriber(topics.relay_status, String, self._relay_cb, queue_size=1)
        self.keepalive = rospy.Timer(rospy.Duration(1.0 / 25.0), self._publish_hold)

    def _relay_cb(self, msg: String) -> None:
        self.relay.set(decode_relay_status(msg.data))

    def _publish_hold(self, _event=None) -> None:
        self.commander.publish_active_target()

    def wait_feedback(self) -> PoseStamped:
        deadline = time.monotonic() + self.system.eef.feedback_wait_timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg = self.pose.get(max_age_s=self.system.eef.max_feedback_age_s)
            joints = self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            )
            if pose_msg_to_xyz_quat(msg) is not None and joints is not None:
                return msg
            time.sleep(0.05)
        raise RuntimeError(
            "No fresh named joint and EEF pose feedback for the IK nudge"
        )

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
        current = self.wait_feedback()
        hold = self.joints.positions(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )
        if hold is None:
            raise RuntimeError("Fresh named joints disappeared before staging hold")
        self.commander.hold_current_target()
        self.commander.publish_active_target()
        wait_for_staged_joint_alignment(
            self.staged,
            hold,
            dof=len(self.system.joint_safety.names),
            timeout_s=self.system.startup.topic_timeout_s,
            max_age_s=self.system.relay.max_input_age_s,
            tolerance_rad=self.system.joint_safety.initial_alignment_tolerance_rad,
            is_shutdown=rospy.is_shutdown,
        )
        info("jointTracker staged a fresh initial hold aligned with joint feedback.")
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
            current = self.wait_feedback()
            xyz, quat = pose_msg_to_xyz_quat(current)
            target_xyz = xyz + delta
            lower = np.asarray(self.system.eef.xyz_min)
            upper = np.asarray(self.system.eef.xyz_max)
            if np.any(target_xyz < lower) or np.any(target_xyz > upper):
                warning(
                    f"Skipped {label}: target_xyz="
                    f"{np.round(target_xyz, 4).tolist()} is outside the tracked "
                    "EEF workspace."
                )
                continue
            action8 = np.concatenate([target_xyz, quat, [0.0]])
            self.commander.publish_action(action8, publish_gripper=False)
            step(
                f"Published IK {label} target_xyz="
                f"{np.round(target_xyz, 4).tolist()} "
                f"delta_cm={np.round(delta * 100.0, 2).tolist()}"
            )
            time.sleep(self.settle_s)
            actual = self.wait_feedback()
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
        description="Step-gated EEF IK nudge through the staged joint runtime."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Enable relay and publish commands."
    )
    parser.add_argument("--config", type=Path, default=ROOT_DIR / DEFAULT_SYSTEM_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
