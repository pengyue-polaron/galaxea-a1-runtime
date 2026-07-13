"""Small, dependency-free URDF forward kinematics for a serial chain."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RevoluteJoint:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class SerialChainFK:
    """Forward kinematics for the unique URDF chain between two links."""

    def __init__(self, joints: Sequence[RevoluteJoint], *, base_link: str, tip_link: str) -> None:
        if not joints:
            raise ValueError("serial chain must contain at least one movable joint")
        self.joints = tuple(joints)
        self.base_link = base_link
        self.tip_link = tip_link

    @classmethod
    def from_urdf(cls, path: Path, *, base_link: str, tip_link: str) -> "SerialChainFK":
        root = ET.parse(path).getroot()
        by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                by_child[_required_attr(child, "link")] = joint

        elements: list[ET.Element] = []
        current = tip_link
        while current != base_link:
            joint = by_child.get(current)
            if joint is None:
                raise ValueError(f"no URDF chain from {base_link!r} to {tip_link!r}")
            elements.append(joint)
            parent = joint.find("parent")
            if parent is None:
                raise ValueError(f"joint {_required_attr(joint, 'name')!r} has no parent")
            current = _required_attr(parent, "link")
        elements.reverse()

        joints: list[RevoluteJoint] = []
        for element in elements:
            joint_type = element.get("type")
            if joint_type == "fixed":
                continue
            if joint_type not in {"revolute", "continuous"}:
                raise ValueError(f"unsupported joint type in serial chain: {joint_type!r}")
            origin = element.find("origin")
            axis = element.find("axis")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"incomplete URDF joint {_required_attr(element, 'name')!r}")
            joints.append(
                RevoluteJoint(
                    name=_required_attr(element, "name"),
                    parent=_required_attr(parent, "link"),
                    child=_required_attr(child, "link"),
                    xyz=_vector_attr(origin, "xyz", "0 0 0"),
                    rpy=_vector_attr(origin, "rpy", "0 0 0"),
                    axis=_vector_attr(axis, "xyz", "0 0 1"),
                )
            )
        return cls(joints, base_link=base_link, tip_link=tip_link)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def pose(self, positions: Sequence[float]) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (len(self.joints),):
            raise ValueError(f"expected {len(self.joints)} joints, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint positions contain non-finite values")

        transform = np.eye(4, dtype=np.float64)
        for position, joint in zip(values, self.joints, strict=True):
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rotation_from_rpy(joint.rpy)
            origin[:3, 3] = joint.xyz
            motion = np.eye(4, dtype=np.float64)
            motion[:3, :3] = _rotation_from_axis_angle(joint.axis, float(position))
            transform = transform @ origin @ motion
        return np.concatenate((transform[:3, 3], _quaternion_from_rotation(transform[:3, :3])))


def relative_pose(target_pose: Sequence[float], initial_pose: Sequence[float]) -> np.ndarray:
    """Return RoboTwin-style pose delta: world translation and local rotation."""

    target = _pose7(target_pose, "target_pose")
    initial = _pose7(initial_pose, "initial_pose")
    delta_rotation = _rotation_from_quaternion(initial[3:7]).T @ _rotation_from_quaternion(target[3:7])
    return np.concatenate((target[:3] - initial[:3], _quaternion_from_rotation(delta_rotation)))


def compose_relative_pose(delta_pose: Sequence[float], initial_pose: Sequence[float]) -> np.ndarray:
    """Compose a RoboTwin-style pose delta onto an episode initial pose."""

    delta = _pose7(delta_pose, "delta_pose")
    initial = _pose7(initial_pose, "initial_pose")
    rotation = _rotation_from_quaternion(initial[3:7]) @ _rotation_from_quaternion(delta[3:7])
    return np.concatenate((initial[:3] + delta[:3], _quaternion_from_rotation(rotation)))


def _pose7(values: Sequence[float], label: str) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite 7-vector")
    pose = pose.copy()
    norm = float(np.linalg.norm(pose[3:7]))
    if norm < 1e-8:
        raise ValueError(f"{label} has an invalid quaternion")
    pose[3:7] /= norm
    return pose


def _required_attr(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if not value:
        raise ValueError(f"URDF <{element.tag}> is missing {name!r}")
    return value


def _vector_attr(element: ET.Element | None, name: str, default: str) -> np.ndarray:
    raw = default if element is None else element.get(name, default)
    values = np.fromstring(raw, dtype=np.float64, sep=" ")
    if values.shape != (3,):
        raise ValueError(f"URDF {name!r} must contain three numbers: {raw!r}")
    return values


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("URDF joint axis must be non-zero")
    x, y, z = axis / norm
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotation_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    # Symmetric eigensystem conversion is stable near 180-degree rotations.
    r = np.asarray(rotation, dtype=np.float64)
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[2, 1] - r[1, 2]],
            [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[0, 2] - r[2, 0]],
            [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
        ],
        dtype=np.float64,
    ) / 3.0
    _, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, -1]
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion
