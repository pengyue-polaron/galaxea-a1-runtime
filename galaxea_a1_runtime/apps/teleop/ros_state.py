"""Fresh ROS state assembly for teleoperation recording."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.apps.eef_bridge import pose_msg_to_xyz_quat
from galaxea_a1_runtime.gripper import normalize_stroke
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache
from galaxea_a1_runtime.runtime.ros_feedback import ordered_joint_positions
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


@dataclass(frozen=True)
class JointSnapshot:
    ros_stamp_s: float
    names: tuple[str, ...]
    positions: tuple[float, ...]


@dataclass(frozen=True)
class TeleopStateSample:
    values: tuple[float, ...]
    ros_stamp_s: float


class RosTeleopState:
    def __init__(self, config: TeleopConfig):
        self.config = config
        self.system = config.system
        self.joints = LatestMessageCache()
        self.eef = LatestMessageCache()
        self.action = LatestMessageCache()
        self.gripper_feedback = LatestMessageCache()
        self.gripper_action = LatestMessageCache()

        rospy.Subscriber(
            self.system.topics.joint_states,
            JointState,
            self.joints.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.system.topics.eef_pose,
            PoseStamped,
            self.eef.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.system.topics.joint_target,
            JointState,
            self.action.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.system.topics.gripper_feedback,
            JointState,
            self.gripper_feedback.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.system.topics.gripper_target,
            gripper_position_control,
            self.gripper_action.callback,
            queue_size=10,
        )

    def wait_ready(self, *, state_mode: StateMode, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if (
                self.state_sample(state_mode) is not None
                and self.action_values() is not None
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(f"ROS state did not become ready within {timeout_s:.1f}s")

    def joint_snapshot(self) -> JointSnapshot | None:
        msg = self.joints.get(max_age_s=self.system.joint_safety.max_feedback_age_s)
        if msg is None:
            return None
        expected = self.system.joint_safety.names
        positions = ordered_joint_positions(
            msg,
            expected,
            label="teleop joint feedback",
        )
        return JointSnapshot(
            ros_stamp_s=_required_stamp_to_seconds(
                getattr(getattr(msg, "header", None), "stamp", None)
            ),
            names=expected,
            positions=positions,
        )

    def eef_vector(self) -> tuple[float, ...] | None:
        msg = self.eef.get(max_age_s=self.system.eef.max_feedback_age_s)
        decoded = pose_msg_to_xyz_quat(msg, min_quat_norm=self.system.eef.min_quat_norm)
        if decoded is None:
            return None
        xyz, quat = decoded
        return tuple(float(value) for value in (*xyz, *quat))

    def state_values(self, mode: StateMode) -> tuple[float, ...] | None:
        sample = self.state_sample(mode)
        return None if sample is None else sample.values

    def state_sample(self, mode: StateMode) -> TeleopStateSample | None:
        joints = self.joint_snapshot()
        if joints is None:
            return None
        joint_values = joints.positions
        gripper = self.gripper_feedback_norm()
        if gripper is None:
            return None
        eef = self.eef_vector()
        if mode == StateMode.EEF:
            if eef is None:
                return None
            values = (*eef, gripper)
            return TeleopStateSample(values, joints.ros_stamp_s)
        if mode == StateMode.JOINT:
            values = (*joint_values, gripper)
            return TeleopStateSample(values, joints.ros_stamp_s)
        if eef is None:
            return None
        values = (*eef, *joint_values, gripper)
        return TeleopStateSample(values, joints.ros_stamp_s)

    def action_values(self) -> tuple[float, ...] | None:
        msg = self.action.get(max_age_s=self.system.joint_safety.max_feedback_age_s)
        if msg is None:
            return None
        target = ordered_joint_positions(
            msg,
            self.system.joint_safety.names,
            label="teleop joint action",
            allow_unnamed=False,
        )
        gripper = self.gripper_action_norm()
        if gripper is None:
            return None
        return (*target, gripper)

    def gripper_feedback_norm(self) -> float | None:
        msg = self.gripper_feedback.get(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )
        if msg is not None and getattr(msg, "position", None):
            return normalize_stroke(
                float(msg.position[0]),
                stroke_min_mm=self.system.gripper.stroke_min_mm,
                stroke_max_mm=self.system.gripper.stroke_max_mm,
            )
        return None

    def gripper_action_norm(self) -> float | None:
        msg = self.gripper_action.get(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )
        if msg is None:
            return None
        stroke = getattr(msg, "gripper_stroke", None)
        if stroke is None:
            return None
        return normalize_stroke(
            float(stroke),
            stroke_min_mm=self.system.gripper.stroke_min_mm,
            stroke_max_mm=self.system.gripper.stroke_max_mm,
        )


def _required_stamp_to_seconds(stamp: Any) -> float:
    if stamp is None:
        raise ValueError("joint feedback has no ROS timestamp")
    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        try:
            value = float(to_sec())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("joint feedback has an invalid ROS timestamp") from exc
    else:
        try:
            value = float(stamp.secs) + float(stamp.nsecs) * 1e-9
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("joint feedback has an invalid ROS timestamp") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"joint feedback has an invalid ROS timestamp: {value!r}")
    return value
