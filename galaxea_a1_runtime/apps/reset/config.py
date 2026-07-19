"""Strict shared A1 reset-pose configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple,
    floating,
    load_toml,
    require_exact_keys,
    required_table,
)
from galaxea_a1_runtime.configuration.system import SystemConfig


@dataclass(frozen=True)
class A1HomeTopics:
    joint_states: str
    target: str
    staged_command: str
    relay_enable: str
    relay_status: str
    gripper_target: str


@dataclass(frozen=True)
class A1HomeMotion:
    hz: float
    min_duration_s: float
    max_velocity_rad_s: float
    tracker_alignment_timeout_s: float
    tracker_alignment_tolerance_rad: float
    relay_enable_timeout_s: float
    max_relay_status_age_s: float
    hold_s: float
    goal_tolerance_rad: float


@dataclass(frozen=True)
class A1GripperHome:
    enabled: bool
    closed_stroke_mm: float
    publish_hz: float
    publish_s: float


@dataclass(frozen=True)
class A1HomePose:
    path: Path
    names: tuple[str, ...]
    positions: tuple[float, ...]
    topics: A1HomeTopics
    motion: A1HomeMotion
    gripper: A1GripperHome


def load_a1_home_pose(
    path: Path,
    *,
    system: SystemConfig,
    repo_root: Path | None = None,
) -> A1HomePose:
    path, _, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"joints", "gripper", "motion"},
        label="A1 reset pose",
    )
    joints = required_table(data, "joints")
    require_exact_keys(joints, required={"position_rad"}, label="reset joints")
    names = system.joint_safety.names
    positions = float_tuple(joints, "position_rad", len(names))
    _validate_joint_targets(
        names,
        positions,
        system.joint_safety.lower_limits,
        system.joint_safety.upper_limits,
    )

    gripper_data = required_table(data, "gripper")
    require_exact_keys(
        gripper_data,
        required={"enabled", "closed_stroke_mm", "publish_hz", "publish_s"},
        label="reset gripper",
    )
    gripper = A1GripperHome(
        enabled=boolean(gripper_data, "enabled"),
        closed_stroke_mm=floating(gripper_data, "closed_stroke_mm"),
        publish_hz=floating(gripper_data, "publish_hz"),
        publish_s=floating(gripper_data, "publish_s"),
    )
    if not (
        system.gripper.stroke_min_mm
        <= gripper.closed_stroke_mm
        <= system.gripper.stroke_max_mm
    ):
        raise ValueError("reset gripper target is outside the system stroke range")

    motion_data = required_table(data, "motion")
    require_exact_keys(
        motion_data,
        required={
            "hz",
            "min_duration_s",
            "max_velocity_rad_s",
            "tracker_alignment_timeout_s",
            "hold_s",
            "goal_tolerance_rad",
        },
        label="reset motion",
    )
    motion = A1HomeMotion(
        hz=floating(motion_data, "hz"),
        min_duration_s=floating(motion_data, "min_duration_s"),
        max_velocity_rad_s=floating(motion_data, "max_velocity_rad_s"),
        tracker_alignment_timeout_s=floating(
            motion_data, "tracker_alignment_timeout_s"
        ),
        tracker_alignment_tolerance_rad=(
            system.joint_safety.initial_alignment_tolerance_rad
        ),
        relay_enable_timeout_s=system.relay.enable_timeout_s,
        max_relay_status_age_s=system.relay.max_status_age_s,
        hold_s=floating(motion_data, "hold_s"),
        goal_tolerance_rad=floating(motion_data, "goal_tolerance_rad"),
    )
    pose = A1HomePose(
        path=path,
        names=names,
        positions=positions,
        topics=A1HomeTopics(
            joint_states=system.topics.joint_states,
            target=system.topics.joint_target,
            staged_command=system.topics.staged_command,
            relay_enable=system.topics.motion_enable,
            relay_status=system.topics.relay_status,
            gripper_target=system.topics.gripper_target,
        ),
        motion=motion,
        gripper=gripper,
    )
    validate_a1_home_pose(pose)
    return pose


def validate_a1_home_pose(pose: A1HomePose) -> None:
    for field, value in pose.motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"motion.{field} must be positive")
    if pose.gripper.publish_hz <= 0 or pose.gripper.publish_s <= 0:
        raise ValueError("gripper publish_hz/publish_s must be positive")


def _validate_joint_targets(
    names: tuple[str, ...],
    positions: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> None:
    violations = [
        f"{name}={value:g} outside [{lo:g}, {hi:g}]"
        for name, value, lo, hi in zip(names, positions, lower, upper, strict=True)
        if value < lo or value > hi
    ]
    if violations:
        raise ValueError(
            "reset joint target violates system limits: " + "; ".join(violations)
        )
