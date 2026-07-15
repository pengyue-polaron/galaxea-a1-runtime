"""Reusable EEF bridge utilities for policy app scripts."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "EefCommandPublisher",
    "condition_state_from_action8",
    "format_xyz_direction",
    "pose_msg_to_xyz_quat",
]


def pose_msg_to_xyz_quat(
    msg: Any, *, min_quat_norm: float = 1e-9
) -> tuple[np.ndarray, np.ndarray] | None:
    if msg is None:
        return None
    position = msg.pose.position
    orientation = msg.pose.orientation
    xyz = np.array([position.x, position.y, position.z], dtype=np.float64)
    quat = np.array(
        [orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64
    )
    norm = np.linalg.norm(quat)
    if norm < min_quat_norm:
        return None
    return xyz, quat / norm


def condition_state_from_action8(
    action8: Sequence[float],
    *,
    frame_chunk_size: int,
    action_per_frame: int,
) -> np.ndarray:
    action = np.asarray(action8, dtype=np.float64).reshape(8)
    return (
        np.broadcast_to(
            action[:, None, None],
            (8, frame_chunk_size, action_per_frame),
        )
        .astype(np.float32)
        .copy()
    )


def format_xyz_direction(delta_xyz: Sequence[float], *, deadband_m: float) -> str:
    parts: list[str] = []
    for axis, value in zip(("x", "y", "z"), delta_xyz, strict=True):
        if abs(float(value)) < deadband_m:
            continue
        parts.append(f"{axis}{'+' if value > 0 else '-'}")
    return ",".join(parts) if parts else "hold"


@dataclass
class EefCommandPublisher:
    """Publish absolute EEF targets and gripper commands through ROS-like objects."""

    rospy: Any
    pose_pub: Any
    gripper_pub: Any
    motion_enable_pub: Any
    pose_msg_type: Any
    bool_msg_type: Any
    gripper_msg_type: Any
    command_frame: str
    gripper_to_stroke: Callable[[float], float]
    execute: bool = True
    active_pose_target: Any | None = None
    active_pose_lock: threading.Lock = field(default_factory=threading.Lock)

    def publish_motion_enable(self, enabled: bool) -> None:
        if not self.execute:
            return
        self.motion_enable_pub.publish(self.bool_msg_type(data=bool(enabled)))

    def set_active_pose_target_from_msg(self, msg: Any) -> None:
        target = self.pose_msg_type()
        target.header.frame_id = self.command_frame
        target.pose = msg.pose
        with self.active_pose_lock:
            self.active_pose_target = target

    def set_active_action(self, action8: Sequence[float]) -> Any:
        action = np.asarray(action8, dtype=np.float64).reshape(8)
        msg = self.pose_msg_type()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = self.command_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
            float, action[:3]
        )
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = map(float, action[3:7])
        with self.active_pose_lock:
            self.active_pose_target = msg
        return msg

    def publish_active_pose_target(self) -> None:
        if not self.execute:
            return
        with self.active_pose_lock:
            if self.active_pose_target is None:
                return
            target = self.pose_msg_type()
            target.header.frame_id = self.active_pose_target.header.frame_id
            target.pose = self.active_pose_target.pose
        target.header.stamp = self.rospy.Time.now()
        self.pose_pub.publish(target)

    def publish_action(
        self, action8: Sequence[float], *, publish_gripper: bool
    ) -> None:
        msg = self.set_active_action(action8)
        self.publish_active_pose_target()
        if not self.execute or not publish_gripper:
            return
        action = np.asarray(action8, dtype=np.float64).reshape(8)
        grip = self.gripper_msg_type()
        grip.header.stamp = msg.header.stamp
        grip.gripper_stroke = self.gripper_to_stroke(float(action[7]))
        self.gripper_pub.publish(grip)
