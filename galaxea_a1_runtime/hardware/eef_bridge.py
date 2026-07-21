"""ROS-facing EEF command and feedback adapters."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from galaxea_a1_runtime.console import info
from galaxea_a1_runtime.hardware.eef_ik import A1EefIkSolver, IkSolution

__all__ = [
    "EefIkCommandPublisher",
    "pose_msg_to_xyz_quat",
]


def pose_msg_to_xyz_quat(
    msg: Any, *, min_quat_norm: float = 1e-9
) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.isfinite(min_quat_norm) or min_quat_norm <= 0:
        raise ValueError("min_quat_norm must be finite and positive")
    if msg is None:
        return None
    try:
        position = msg.pose.position
        orientation = msg.pose.orientation
        xyz = np.array([position.x, position.y, position.z], dtype=np.float64)
        quat = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < min_quat_norm:
        return None
    return xyz, quat / norm


@dataclass
class EefIkCommandPublisher:
    """Solve absolute EEF targets and publish named joint targets."""

    rospy: Any
    target_pub: Any
    gripper_pub: Any
    motion_enable_pub: Any
    joint_state_msg_type: Any
    bool_msg_type: Any
    gripper_msg_type: Any
    joint_names: tuple[str, ...]
    current_joint_positions: Callable[[], Sequence[float] | None]
    solver: A1EefIkSolver
    gripper_to_stroke: Callable[[float], float]
    execute: bool
    log_solutions: bool = True
    active_joint_target: Any | None = None
    active_target_lock: threading.Lock = field(default_factory=threading.Lock)

    def publish_motion_enable(self, enabled: bool) -> None:
        if not self.execute:
            return
        self.motion_enable_pub.publish(self.bool_msg_type(data=bool(enabled)))

    def hold_current_target(self) -> tuple[float, ...]:
        current = self.current_joint_positions()
        if current is None:
            raise RuntimeError("Cannot stage an IK hold without fresh joint feedback")
        return self._set_active_joint_target(current)

    def set_active_action(self, action8: Sequence[float]) -> IkSolution:
        action = np.asarray(action8, dtype=np.float64).reshape(8)
        if not np.all(np.isfinite(action)):
            raise ValueError("EEF IK action must contain only finite values")
        current = self.current_joint_positions()
        if current is None:
            raise RuntimeError("Cannot solve EEF IK without fresh joint feedback")
        solution = self.solver.solve(current, action[:3], action[3:7])
        self._set_active_joint_target(solution.joint_positions)
        if self.log_solutions:
            info(
                "EEF IK solved: "
                f"iterations={solution.iterations} "
                f"position_error_mm={solution.position_error_m * 1000.0:.3f} "
                f"orientation_error_deg="
                f"{np.degrees(solution.orientation_error_rad):.3f} "
                f"max_joint_delta_rad={solution.max_joint_delta_rad:.4f}"
            )
        return solution

    def publish_active_target(self) -> None:
        if not self.execute:
            return
        with self.active_target_lock:
            if self.active_joint_target is None:
                return
            target = self.joint_state_msg_type()
            target.name = list(self.active_joint_target.name)
            target.position = list(self.active_joint_target.position)
        target.header.stamp = self.rospy.Time.now()
        self.target_pub.publish(target)

    def publish_action(
        self, action8: Sequence[float], *, publish_gripper: bool
    ) -> None:
        action = np.asarray(action8, dtype=np.float64).reshape(8)
        self.set_active_action(action)
        self.publish_active_target()
        if publish_gripper:
            self.publish_gripper(float(action[7]))

    def publish_gripper(self, gripper_norm: float) -> None:
        if not self.execute:
            return
        grip = self.gripper_msg_type()
        grip.header.stamp = self.rospy.Time.now()
        grip.gripper_stroke = self.gripper_to_stroke(gripper_norm)
        self.gripper_pub.publish(grip)

    def _set_active_joint_target(self, positions: Sequence[float]) -> tuple[float, ...]:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.shape != (len(self.joint_names),) or not np.all(np.isfinite(values)):
            raise ValueError("IK joint target has invalid shape or values")
        ordered = tuple(float(value) for value in values)
        target = self.joint_state_msg_type()
        target.name = list(self.joint_names)
        target.position = list(ordered)
        with self.active_target_lock:
            self.active_joint_target = target
        return ordered
