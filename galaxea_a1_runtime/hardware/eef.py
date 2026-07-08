"""Pure end-effector pose helpers for hardware adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
from typing import Sequence

from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import ActionMode


@dataclass(frozen=True)
class EefPose:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    frame_id: str = "world"

    def normalized(self) -> "EefPose":
        return EefPose(
            xyz=_vector(self.xyz, 3, "xyz"),
            quat_xyzw=normalize_quat(self.quat_xyzw),
            frame_id=self.frame_id,
        )


def action_to_eef_target(current: EefPose, action: RuntimeAction) -> EefPose | None:
    """Convert a normalized runtime EEF action into an absolute target pose.

    Returns `None` when the action contains no arm motion and only addresses the
    gripper. Joint-space actions are intentionally not supported by the safe EEF
    adapter.
    """

    values = action.as_dict()
    if action.mode not in (ActionMode.EEF_TRANSLATION, ActionMode.EEF_DELTA):
        raise ValueError(f"ROS1 safe EEF adapter does not support action mode {action.mode}")

    delta_xyz = (
        float(values.get("delta_x", 0.0)),
        float(values.get("delta_y", 0.0)),
        float(values.get("delta_z", 0.0)),
    )
    if not any(abs(v) > 1e-12 for v in delta_xyz) and not _has_rotation_delta(values):
        return None

    pose = current.normalized()
    target_xyz = tuple(pose.xyz[i] + delta_xyz[i] for i in range(3))
    target_quat = pose.quat_xyzw
    if action.mode == ActionMode.EEF_DELTA:
        delta_quat = quat_from_rpy(
            float(values.get("delta_roll", 0.0)),
            float(values.get("delta_pitch", 0.0)),
            float(values.get("delta_yaw", 0.0)),
        )
        target_quat = normalize_quat(quat_multiply(target_quat, delta_quat))
    return EefPose(
        xyz=target_xyz,
        quat_xyzw=target_quat,
        frame_id=pose.frame_id,
    )


def normalize_quat(quat_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    values = _vector(quat_xyzw, 4, "quat_xyzw")
    norm = sqrt(sum(v * v for v in values))
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    return tuple(v / norm for v in values)  # type: ignore[return-value]


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    for label, value in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
        if not isfinite(value):
            raise ValueError(f"{label} must be finite, got {value!r}")
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    return normalize_quat(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def quat_multiply(
    left_xyzw: Sequence[float],
    right_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = normalize_quat(left_xyzw)
    rx, ry, rz, rw = normalize_quat(right_xyzw)
    return normalize_quat(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _has_rotation_delta(values: dict[str, float]) -> bool:
    return any(
        abs(float(values.get(name, 0.0))) > 1e-12
        for name in ("delta_roll", "delta_pitch", "delta_yaw")
    )


def _vector(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{label} must have {size} values, got {len(values)}")
    result = tuple(float(v) for v in values)
    if not all(isfinite(v) for v in result):
        raise ValueError(f"{label} must be finite, got {values!r}")
    return result
