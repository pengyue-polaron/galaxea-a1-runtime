"""ROS adapter for the shared, staged A1 reset."""

from __future__ import annotations

import math
import time
from typing import Any

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.reset.config import A1HomePose
from galaxea_a1_runtime.apps.reset.progress import ResetProgress
from galaxea_a1_runtime.runtime.relay import RelayMonitor
from galaxea_a1_runtime.runtime.ros_feedback import (
    A1JointStateCache,
    StagedCommandMonitor,
)


class A1HomeRunner:
    def __init__(self, pose: A1HomePose, progress: ResetProgress):
        self.pose = pose
        self.progress = progress
        self.joints = A1JointStateCache(pose.names)
        self.staged = StagedCommandMonitor()
        self.relay = RelayMonitor(pose.motion.max_relay_status_age_s)
        rospy.init_node("a1_return_home", anonymous=False, disable_signals=True)
        self.target_pub = rospy.Publisher(pose.topics.target, JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher(
            pose.topics.gripper_target,
            gripper_position_control,
            queue_size=1,
        )
        self.enable_pub = rospy.Publisher(
            pose.topics.relay_enable, Bool, queue_size=1, latch=True
        )
        rospy.Subscriber(
            pose.topics.joint_states, JointState, self.joints.callback, queue_size=1
        )
        rospy.Subscriber(
            pose.topics.staged_command,
            arm_control,
            self.staged.callback,
            queue_size=1,
        )
        rospy.Subscriber(
            pose.topics.relay_status, String, self.relay.callback, queue_size=1
        )

    def run(self) -> None:
        motion = self.pose.motion
        rate = rospy.Rate(motion.hz)
        self.enable_pub.publish(Bool(data=False))
        try:
            current = self.wait_for_joints(timeout_s=motion.tracker_alignment_timeout_s)
            self.progress.update("A1", 0)
            self.wait_for_staged_alignment(
                current,
                timeout_s=motion.tracker_alignment_timeout_s,
                tolerance_rad=motion.tracker_alignment_tolerance_rad,
                rate=rate,
            )
            self.enable_pub.publish(Bool(data=True))
            self.wait_for_relay_active(timeout_s=motion.relay_enable_timeout_s)
            self.close_gripper()
            self.move_smooth(current, rate)
            final = self.hold_target(rate)
            self.close_gripper()
            error = max_vector_error(final, self.pose.positions)
            if error > motion.goal_tolerance_rad:
                raise RuntimeError(
                    f"Reset pose error {error:.4f} rad exceeds tolerance "
                    f"{motion.goal_tolerance_rad:.4f} rad"
                )
            self.progress.update("A1", 100)
        finally:
            self.enable_pub.publish(Bool(data=False))
            time.sleep(0.2)

    def wait_for_joints(self, *, timeout_s: float) -> tuple[float, ...]:
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions = self.joints.positions()
            if positions is not None and all(
                math.isfinite(value) for value in positions
            ):
                return positions
            time.sleep(0.05)
        raise RuntimeError(
            f"Reset has no usable joint feedback on {self.pose.topics.joint_states}"
        )

    def wait_for_staged_alignment(
        self,
        target: tuple[float, ...],
        *,
        timeout_s: float,
        tolerance_rad: float,
        rate: Any,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: float | None = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.publish_target(target)
            last_error = self.staged.max_error(target, len(target))
            if last_error is not None and last_error <= tolerance_rad:
                return
            rate.sleep()
        detail = (
            "no staged command"
            if last_error is None
            else f"last error {last_error:.4f} rad"
        )
        raise RuntimeError(
            f"Tracker did not align within {timeout_s:.1f}s "
            f"({detail}, tolerance {tolerance_rad:.4f} rad)"
        )

    def wait_for_relay_active(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.relay.is_active():
                return
            status, _ = self.relay.status()
            if status is not None and status.state == "FAULT":
                break
            time.sleep(0.05)
        raise RuntimeError(f"Relay did not reach ACTIVE: {self.relay.summary()}")

    def move_smooth(self, start: tuple[float, ...], rate: Any) -> None:
        target = self.pose.positions
        motion = self.pose.motion
        duration_s = max(
            motion.min_duration_s,
            max_vector_error(start, target) / motion.max_velocity_rad_s,
        )
        steps = max(1, int(duration_s * motion.hz))
        for step in range(steps + 1):
            alpha = step / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            command = tuple(
                start[index] + (target[index] - start[index]) * smooth
                for index in range(len(target))
            )
            self.publish_target(command)
            self.progress.update("A1", alpha * 100.0)
            if not self.relay.is_active():
                raise RuntimeError(
                    f"Relay left ACTIVE while homing: {self.relay.summary()}"
                )
            rate.sleep()

    def hold_target(self, rate: Any) -> tuple[float, ...]:
        deadline = time.monotonic() + self.pose.motion.hold_s
        final = self.wait_for_joints(timeout_s=1.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.publish_target(self.pose.positions)
            final = self.wait_for_joints(timeout_s=0.5)
            if (
                max_vector_error(final, self.pose.positions)
                <= self.pose.motion.goal_tolerance_rad
            ):
                return final
            rate.sleep()
        return final

    def publish_target(self, target: tuple[float, ...]) -> None:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.pose.names)
        msg.position = list(target)
        self.target_pub.publish(msg)

    def close_gripper(self) -> None:
        gripper = self.pose.gripper
        if not gripper.enabled:
            return
        rate = rospy.Rate(gripper.publish_hz)
        deadline = time.monotonic() + gripper.publish_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg = gripper_position_control()
            msg.header.stamp = rospy.Time.now()
            msg.gripper_stroke = float(gripper.closed_stroke_mm)
            self.gripper_pub.publish(msg)
            rate.sleep()


def max_vector_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} != {len(right)}")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))
