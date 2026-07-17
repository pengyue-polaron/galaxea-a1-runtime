"""Pure URDF forward/inverse kinematics for the Galaxea A1 arm."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from galaxea_a1_runtime.configuration.system import SystemConfig


@dataclass(frozen=True)
class IkSolution:
    joint_positions: tuple[float, ...]
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    max_joint_delta_rad: float


@dataclass(frozen=True)
class _RevoluteJoint:
    origin: np.ndarray
    axis: np.ndarray


class A1EefIkSolver:
    """Solve bounded six-DOF EEF targets from the tracked A1 URDF."""

    def __init__(
        self,
        *,
        urdf_path: Path,
        joint_names: Sequence[str],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        max_iterations: int,
        damping: float,
        orientation_weight: float,
        max_iteration_step_rad: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_solution_delta_rad: float,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.lower_limits = _finite_vector(lower_limits, len(self.joint_names), "lower")
        self.upper_limits = _finite_vector(upper_limits, len(self.joint_names), "upper")
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("IK lower joint limits must be below upper limits")
        if isinstance(max_iterations, bool) or max_iterations <= 0:
            raise ValueError("IK max_iterations must be a positive integer")
        numeric = {
            "damping": damping,
            "orientation_weight": orientation_weight,
            "max_iteration_step_rad": max_iteration_step_rad,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "max_solution_delta_rad": max_solution_delta_rad,
        }
        if any(not math.isfinite(value) or value <= 0 for value in numeric.values()):
            raise ValueError(f"IK settings must be finite and positive: {numeric}")
        self.max_iterations = max_iterations
        self.damping = float(damping)
        self.orientation_weight = float(orientation_weight)
        self.max_iteration_step_rad = float(max_iteration_step_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.max_solution_delta_rad = float(max_solution_delta_rad)
        self.joints = _load_chain(urdf_path, self.joint_names)

    def forward(
        self, joint_positions: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        transform, _, _ = self._kinematics(joint_positions)
        return transform[:3, 3].copy(), _matrix_to_quat(transform[:3, :3])

    def solve(
        self,
        current_joint_positions: Sequence[float],
        target_xyz: Sequence[float],
        target_quat_xyzw: Sequence[float],
    ) -> IkSolution:
        start = _finite_vector(
            current_joint_positions, len(self.joints), "current joint positions"
        )
        if np.any(start < self.lower_limits) or np.any(start > self.upper_limits):
            raise ValueError("current joint positions violate tracked limits")
        target_position = _finite_vector(target_xyz, 3, "target xyz")
        target_rotation = _quat_to_matrix(target_quat_xyzw)
        joints = start.copy()
        position_error = float("inf")
        orientation_error = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            transform, origins, axes = self._kinematics(joints)
            position_delta = target_position - transform[:3, 3]
            orientation_delta = _rotation_vector(target_rotation @ transform[:3, :3].T)
            position_error = float(np.linalg.norm(position_delta))
            orientation_error = float(np.linalg.norm(orientation_delta))
            if (
                position_error <= self.position_tolerance_m
                and orientation_error <= self.orientation_tolerance_rad
            ):
                break

            jacobian = np.empty((6, len(self.joints)), dtype=np.float64)
            for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
                jacobian[:3, index] = np.cross(axis, transform[:3, 3] - origin)
                jacobian[3:, index] = axis
            weighted = jacobian.copy()
            weighted[3:] *= self.orientation_weight
            error = np.concatenate(
                [position_delta, self.orientation_weight * orientation_delta]
            )
            normal = weighted @ weighted.T
            normal += (self.damping**2) * np.eye(6, dtype=np.float64)
            delta = weighted.T @ np.linalg.solve(normal, error)
            largest = float(np.max(np.abs(delta)))
            if largest > self.max_iteration_step_rad:
                delta *= self.max_iteration_step_rad / largest
            joints = np.clip(
                joints + delta,
                self.lower_limits,
                self.upper_limits,
            )
        else:
            iteration = self.max_iterations

        transform, _, _ = self._kinematics(joints)
        position_error = float(np.linalg.norm(target_position - transform[:3, 3]))
        orientation_error = float(
            np.linalg.norm(_rotation_vector(target_rotation @ transform[:3, :3].T))
        )
        if (
            position_error > self.position_tolerance_m
            or orientation_error > self.orientation_tolerance_rad
        ):
            raise RuntimeError(
                "A1 EEF IK did not converge: "
                f"iterations={iteration} position_error_m={position_error:.6f} "
                f"orientation_error_rad={orientation_error:.6f}"
            )
        max_delta = float(np.max(np.abs(joints - start)))
        if max_delta > self.max_solution_delta_rad:
            raise RuntimeError(
                "A1 EEF IK solution exceeds the configured joint delta: "
                f"{max_delta:.6f} > {self.max_solution_delta_rad:.6f} rad"
            )
        return IkSolution(
            joint_positions=tuple(float(value) for value in joints),
            iterations=iteration,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            max_joint_delta_rad=max_delta,
        )

    def _kinematics(
        self, joint_positions: Sequence[float]
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        values = _finite_vector(joint_positions, len(self.joints), "joint positions")
        transform = np.eye(4, dtype=np.float64)
        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for joint, angle in zip(self.joints, values, strict=True):
            transform = transform @ joint.origin
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ joint.axis)
            rotation = np.eye(4, dtype=np.float64)
            rotation[:3, :3] = _axis_angle_matrix(joint.axis, float(angle))
            transform = transform @ rotation
        return transform, tuple(origins), tuple(axes)


def build_eef_ik_solver(system: SystemConfig) -> A1EefIkSolver:
    config = system.eef_ik
    joints = system.joint_safety
    return A1EefIkSolver(
        urdf_path=config.urdf,
        joint_names=joints.names,
        lower_limits=joints.lower_limits,
        upper_limits=joints.upper_limits,
        max_iterations=config.max_iterations,
        damping=config.damping,
        orientation_weight=config.orientation_weight,
        max_iteration_step_rad=config.max_iteration_step_rad,
        position_tolerance_m=config.position_tolerance_m,
        orientation_tolerance_rad=config.orientation_tolerance_rad,
        max_solution_delta_rad=config.max_solution_delta_rad,
    )


def _load_chain(
    urdf_path: Path,
    joint_names: tuple[str, ...],
) -> tuple[_RevoluteJoint, ...]:
    path = urdf_path.expanduser().resolve()
    root = ET.parse(path).getroot()
    by_name = {
        element.attrib.get("name", ""): element for element in root.findall("joint")
    }
    joints: list[_RevoluteJoint] = []
    for name in joint_names:
        element = by_name.get(name)
        if element is None or element.attrib.get("type") != "revolute":
            raise ValueError(f"URDF is missing revolute joint {name!r}")
        origin_element = element.find("origin")
        xyz = _attribute_vector(origin_element, "xyz", default="0 0 0")
        rpy = _attribute_vector(origin_element, "rpy", default="0 0 0")
        origin = np.eye(4, dtype=np.float64)
        origin[:3, :3] = _rpy_matrix(rpy)
        origin[:3, 3] = xyz
        axis_element = element.find("axis")
        if axis_element is None:
            raise ValueError(f"URDF joint {name!r} has no axis")
        axis = _finite_vector(axis_element.attrib.get("xyz", "").split(), 3, name)
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            raise ValueError(f"URDF joint {name!r} has a zero axis")
        axis /= norm
        limit = element.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name!r} has no limits")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"URDF joint {name!r} has invalid limits")
        joints.append(_RevoluteJoint(origin=origin, axis=axis))
    return tuple(joints)


def _attribute_vector(
    element: ET.Element | None, name: str, *, default: str
) -> np.ndarray:
    value = default if element is None else element.attrib.get(name, default)
    return _finite_vector(value.split(), 3, f"URDF {name}")


def _finite_vector(values: Sequence[float], length: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result.copy()


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    cosine, sine = math.cos(angle), math.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _finite_vector(quat_xyzw, 4, "target quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("target quaternion has zero norm")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
        w = 0.25 * scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = (
                math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = (
                math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = (
                math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    return angle / (2.0 * math.sin(angle)) * skew
