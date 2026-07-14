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
from galaxea_a1_runtime.gripper import normalize_stroke
from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


@dataclass(frozen=True)
class JointSnapshot:
    ros_stamp_s: float
    names: tuple[str, ...]
    positions: tuple[float, ...]


class RosTeleopState:
    def __init__(self, args):
        self.args = args
        self.joints = LatestMessageCache()
        self.eef = LatestMessageCache()
        self.action = LatestMessageCache()
        self.gripper_feedback = LatestMessageCache()
        self.gripper_action = LatestMessageCache()

        rospy.Subscriber(
            args.joint_topic, JointState, self.joints.callback, queue_size=10
        )
        rospy.Subscriber(args.eef_topic, PoseStamped, self.eef.callback, queue_size=10)
        rospy.Subscriber(
            args.action_topic, JointState, self.action.callback, queue_size=10
        )
        rospy.Subscriber(
            args.gripper_feedback_topic,
            JointState,
            self.gripper_feedback.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            args.gripper_action_topic,
            gripper_position_control,
            self.gripper_action.callback,
            queue_size=10,
        )

    def wait_ready(self, *, state_mode: StateMode, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            if self.joint_snapshot() is None:
                time.sleep(0.05)
                continue
            if self.action_values() is None:
                time.sleep(0.05)
                continue
            if (
                state_mode in (StateMode.EEF, StateMode.EEF_JOINT)
                and self.eef_vector() is None
            ):
                time.sleep(0.05)
                continue
            return
        raise RuntimeError(f"ROS state did not become ready within {timeout_s:.1f}s")

    def joint_snapshot(self) -> JointSnapshot | None:
        msg = self.joints.get(max_age_s=self.args.max_joint_feedback_age_s)
        if msg is None:
            return None
        names = tuple(str(name) for name in getattr(msg, "name", ()))
        positions = tuple(float(value) for value in getattr(msg, "position", ()))
        if not positions:
            return None
        usable = min(len(names), len(positions)) if names else len(positions)
        return JointSnapshot(
            ros_stamp_s=_stamp_to_seconds(
                getattr(getattr(msg, "header", None), "stamp", None)
            ),
            names=names[:usable]
            if names
            else tuple(f"joint_{i + 1}" for i in range(usable)),
            positions=positions[:usable],
        )

    def eef_vector(self) -> tuple[float, ...] | None:
        msg = self.eef.get(max_age_s=self.args.max_eef_feedback_age_s)
        if msg is None:
            return None
        pose = msg.pose
        quat = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        norm = float(np.linalg.norm(np.asarray(quat, dtype=np.float64)))
        if norm < 1e-9:
            return None
        quat = tuple(value / norm for value in quat)
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            *quat,
        )

    def state_values(self, mode: StateMode) -> tuple[float, ...] | None:
        joints = self.joint_snapshot()
        if joints is None:
            return None
        joint_values = _first_n(joints.positions, 6, label="joint state")
        gripper = self.gripper_feedback_norm()
        if gripper is None:
            return None
        eef = self.eef_vector()
        if mode == StateMode.EEF:
            if eef is None:
                return None
            return (*eef, gripper)
        if mode == StateMode.JOINT:
            return (*joint_values, gripper)
        if eef is None:
            return None
        return (*eef, *joint_values, gripper)

    def action_values(self) -> tuple[float, ...] | None:
        msg = self.action.get(max_age_s=self.args.max_action_age_s)
        if msg is None:
            return None
        positions = tuple(float(value) for value in getattr(msg, "position", ()))
        target = _first_n(positions, 6, label="teleop action")
        gripper = self.gripper_action_norm()
        if gripper is None:
            return None
        return (*target, gripper)

    def gripper_feedback_norm(self) -> float | None:
        msg = self.gripper_feedback.get(max_age_s=self.args.max_gripper_age_s)
        if msg is not None and getattr(msg, "position", None):
            return normalize_stroke(
                float(msg.position[0]),
                stroke_min_mm=self.args.gripper_stroke_min,
                stroke_max_mm=self.args.gripper_stroke_max,
            )
        return None

    def gripper_action_norm(self) -> float | None:
        msg = self.gripper_action.get(max_age_s=self.args.max_gripper_age_s)
        if msg is None:
            return None
        stroke = getattr(msg, "gripper_stroke", None)
        if stroke is None:
            return None
        return normalize_stroke(
            float(stroke),
            stroke_min_mm=self.args.gripper_stroke_min,
            stroke_max_mm=self.args.gripper_stroke_max,
        )

    def ros_stamp(self) -> float:
        joint = self.joint_snapshot()
        if joint is None:
            raise RuntimeError("joint feedback became stale while recording the frame")
        return joint.ros_stamp_s


def _stamp_to_seconds(stamp: Any) -> float:
    if stamp is None:
        return 0.0
    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        try:
            return float(to_sec())
        except Exception:
            return 0.0
    return float(getattr(stamp, "secs", 0)) + float(getattr(stamp, "nsecs", 0)) * 1e-9


def _first_n(values: tuple[float, ...], count: int, *, label: str) -> tuple[float, ...]:
    if len(values) < count:
        raise RuntimeError(f"{label} has {len(values)} values, need {count}")
    return tuple(float(value) for value in values[:count])
