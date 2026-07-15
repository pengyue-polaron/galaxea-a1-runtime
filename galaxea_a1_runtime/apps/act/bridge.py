#!/usr/bin/env python3
# ruff: noqa: E402
"""ACT joint-state policy bridge implementation for Galaxea A1.

The bridge is dry-run by default. When --execute is enabled it publishes only
to the safe joint target topic, then relies on the isolated jointTracker and
safe relay to reach /arm_joint_command_host.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR, include_system_site=False, remove_ros2=True)

import numpy as np

import rospy
from galaxea_a1_runtime.gripper import denormalize_stroke
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String


from galaxea_a1_runtime.apps.act.actions import ActActionValidator
from galaxea_a1_runtime.apps.act.policy import ActPolicyRunner, log
from galaxea_a1_runtime.apps.policy_camera import (
    PolicyCameraSession,
    required_square_front_roi,
)
from galaxea_a1_runtime.runtime.relay import RelayMonitor
from galaxea_a1_runtime.runtime.ros_feedback import (
    A1JointStateCache,
    GripperFeedbackCache,
    StagedCommandMonitor,
)


class ActJointBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.target_names = tuple(args.target_joint_names)
        self.motion_enabled = False
        self.front_roi = required_square_front_roi(args)
        self.action_validator = ActActionValidator(
            joint_names=self.target_names,
            lower_limits=np.asarray(args.lower_limits, dtype=np.float64),
            upper_limits=np.asarray(args.upper_limits, dtype=np.float64),
            execute_steps=int(args.execute_steps_per_inference),
            step_guard_enabled=bool(args.action_step_guard_enabled),
            max_first_delta_rad=float(args.max_first_target_delta_rad),
            max_step_rad=float(args.max_joint_action_step_rad),
        )

        rospy.init_node(
            "act_joint_policy_bridge", anonymous=False, disable_signals=True
        )
        self.joints = A1JointStateCache(self.target_names)
        self.gripper_feedback = GripperFeedbackCache()
        self.relay = RelayMonitor(args.max_relay_status_age)
        self.staged = StagedCommandMonitor()

        rospy.Subscriber(
            args.joint_states_topic, JointState, self.joints.callback, queue_size=1
        )
        rospy.Subscriber(
            args.gripper_feedback_topic,
            JointState,
            self.gripper_feedback.callback,
            queue_size=1,
        )
        rospy.Subscriber(
            args.relay_status_topic, String, self.relay.callback, queue_size=1
        )
        rospy.Subscriber(
            args.staged_command_topic, arm_control, self.staged.callback, queue_size=1
        )
        self.target_pub = rospy.Publisher(args.target_topic, JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            args.gripper_target_topic, gripper_position_control, queue_size=10
        )
        self.motion_enable_pub = rospy.Publisher(
            args.motion_enable_topic, Bool, queue_size=1, latch=True
        )

        self.policy = ActPolicyRunner(args)
        self.cameras = PolicyCameraSession(args, self.front_roi)

    def close(self) -> None:
        if self.motion_enabled:
            self.motion_enable_pub.publish(Bool(data=False))
            self.motion_enabled = False
        if getattr(self, "cameras", None) is not None:
            self.cameras.close()

    def run(self) -> None:
        mode = "EXECUTE" if self.args.execute else "DRY-RUN"
        log(f"[ACT] Bridge started in {mode}. step_mode={self.args.step_mode}")
        model_calls = 0
        while not rospy.is_shutdown():
            if self.args.max_model_calls and model_calls >= self.args.max_model_calls:
                log("[ACT] max_model_calls reached; exiting.")
                return
            if self.args.step_mode and not self._wait_for_operator(model_calls + 1):
                return

            front_bgr, wrist_bgr, state7, current_joints = self._read_observation()
            chunk = self.policy.predict_chunk(
                front_bgr=front_bgr,
                wrist_bgr=wrist_bgr,
                state7=state7,
            )
            model_calls += 1
            self._print_preview(model_calls, chunk, current_joints)

            if not self.args.execute:
                if not self.args.step_mode:
                    time.sleep(1.0)
                continue

            try:
                steps = self.action_validator.validate(chunk, current_joints)
            except RuntimeError as exc:
                self._skip_execution(str(exc))
                continue
            if not self.motion_enabled:
                self._wait_for_staged_alignment(current_joints)
                self._enable_motion()
            self._execute_steps(steps)

    def _wait_for_operator(self, call_index: int) -> bool:
        log(
            f"\n[ACT] INFERENCE #{call_index} READY. "
            "Press Enter to run one new model inference; q=quit."
        )
        try:
            value = input().strip().lower()
        except EOFError:
            value = "q"
        return value not in {"q", "quit", "exit"}

    def _read_observation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], tuple[float, ...]]:
        current_joints = self._wait_for_joints()
        gripper = self._gripper_normalized()
        front_bgr, wrist_bgr = self._wait_for_cameras()
        state7 = (*current_joints, gripper)
        return front_bgr, wrist_bgr, state7, current_joints

    def _wait_for_joints(self) -> tuple[float, ...]:
        deadline = time.monotonic() + self.args.state_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions = self.joints.positions(max_age_s=self.args.max_feedback_age)
            if positions is not None and all(np.isfinite(positions)):
                return positions
            time.sleep(0.02)
        raise RuntimeError(
            f"No fresh usable joint feedback on {self.args.joint_states_topic}"
        )

    def _gripper_normalized(self) -> float:
        feedback = self.gripper_feedback.normalized(
            max_age_s=self.args.max_feedback_age,
            stroke_min_mm=self.args.gripper_stroke_min,
            stroke_max_mm=self.args.gripper_stroke_max,
        )
        if feedback is not None:
            return feedback
        raise RuntimeError(
            f"No fresh gripper feedback on {self.args.gripper_feedback_topic}"
        )

    def _wait_for_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cameras.wait_pair(
            timeout_s=self.args.state_timeout,
            is_shutdown=rospy.is_shutdown,
        )

    def _skip_execution(self, reason: str) -> None:
        self.motion_enable_pub.publish(Bool(data=False))
        self.motion_enabled = False
        log(f"[ACT safety] {reason}")
        log(
            "[ACT safety] Skipping this action; relay is locked/disabled. Press Enter to infer again, q=quit."
        )

    def _print_preview(
        self, call_index: int, chunk: np.ndarray, current_joints: tuple[float, ...]
    ) -> None:
        if not self.args.print_actions:
            return
        count = min(int(self.args.preview_steps), len(chunk))
        first_delta = float(
            np.max(np.abs(chunk[0, :6] - np.asarray(current_joints, dtype=np.float64)))
        )
        adjacent = np.diff(chunk[: max(count, 2), :6], axis=0)
        max_step = float(np.max(np.abs(adjacent))) if adjacent.size else 0.0
        log(
            f"[ACT #{call_index}] first_delta={first_delta:.4f} max_preview_step={max_step:.4f} "
            f"gripper0={chunk[0, 6]:.3f}"
        )
        for idx in range(count):
            row = chunk[idx]
            log(
                f"  step {idx:02d}: joints={np.round(row[:6], 4).tolist()} "
                f"gripper_norm={row[6]:.3f}"
            )

    def _wait_for_staged_alignment(self, target: tuple[float, ...]) -> None:
        rate = rospy.Rate(min(float(self.args.control_hz), 30.0))
        deadline = time.monotonic() + self.args.state_timeout
        last_error: float | None = None
        log(
            "[ACT] Aligning jointTracker staged output with current feedback before relay enable..."
        )
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._publish_joint_target(target)
            last_error = self.staged.max_error(target, len(target))
            if (
                last_error is not None
                and last_error <= self.args.initial_alignment_tolerance
            ):
                log(f"[ACT] Tracker aligned; staged max error={last_error:.4f} rad")
                return
            rate.sleep()
        detail = (
            "no staged command"
            if last_error is None
            else f"last max error {last_error:.4f} rad"
        )
        raise RuntimeError(
            "jointTracker staged output did not align with current target "
            f"({detail}, tolerance={self.args.initial_alignment_tolerance:.4f})"
        )

    def _enable_motion(self) -> None:
        self.motion_enable_pub.publish(Bool(data=True))
        deadline = time.monotonic() + self.args.relay_enable_timeout
        last = self.relay.summary()
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            last = self.relay.summary()
            if self.relay.is_active():
                self.motion_enabled = True
                log("[ACT] Relay ACTIVE; joint targets may move the arm.")
                return
            status, _ = self.relay.status()
            if status is not None and status.state == "FAULT":
                break
            time.sleep(0.05)
        self.motion_enable_pub.publish(Bool(data=False))
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last}")

    def _require_relay_active(self) -> None:
        if self.relay.is_active():
            return
        self.motion_enable_pub.publish(Bool(data=False))
        self.motion_enabled = False
        raise RuntimeError(
            f"A1 relay is not ACTIVE; refusing to publish. Last state: {self.relay.summary()}"
        )

    def _execute_steps(self, steps: np.ndarray) -> None:
        rate = rospy.Rate(float(self.args.control_hz))
        for index, row in enumerate(steps):
            self._require_relay_active()
            stamp = self._publish_joint_target(tuple(float(v) for v in row[:6]))
            gripper_mm = self._publish_gripper(float(row[6]), stamp)
            if self.args.print_actions:
                log(
                    f"[ACT execute] step={index + 1}/{len(steps)} "
                    f"target={np.round(row[:6], 4).tolist()} "
                    f"gripper={row[6]:.3f} gripper_mm={gripper_mm:.1f}"
                )
            rate.sleep()

    def _publish_joint_target(self, target: Sequence[float]) -> Any:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.target_names)
        msg.position = [float(value) for value in target]
        self.target_pub.publish(msg)
        return msg.header.stamp

    def _publish_gripper(self, value: float, stamp: Any) -> float:
        msg = gripper_position_control()
        msg.header.stamp = stamp
        stroke = denormalize_stroke(
            value,
            stroke_min_mm=self.args.gripper_stroke_min,
            stroke_max_mm=self.args.gripper_stroke_max,
        )
        msg.gripper_stroke = stroke
        self.gripper_pub.publish(msg)
        return stroke
