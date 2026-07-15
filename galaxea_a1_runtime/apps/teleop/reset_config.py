"""Strict tracked dual-device reset configuration."""

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
    string_tuple,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig, TeleopLeaderConfig


ROOT_DIR = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class HomeTopics:
    joint_states: str
    target: str
    staged_command: str
    relay_enable: str
    relay_status: str
    gripper_target: str


@dataclass(frozen=True)
class HomeMotion:
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
class LeaderHome:
    enabled: bool
    config: TeleopLeaderConfig
    action: dict[str, float]


@dataclass(frozen=True)
class LeaderMotion:
    hz: float
    min_duration_s: float
    max_velocity_units_s: float
    hold_s: float
    goal_tolerance_units: float
    gripper_goal_tolerance_units: float


@dataclass(frozen=True)
class HomePose:
    path: Path
    names: tuple[str, ...]
    positions: tuple[float, ...]
    topics: HomeTopics
    motion: HomeMotion
    gripper: A1GripperHome
    leader: LeaderHome
    leader_motion: LeaderMotion


def load_home_pose(path: Path, *, teleop: TeleopConfig) -> HomePose:
    path, _, data = load_toml(path, repo_root=ROOT_DIR)
    require_exact_keys(
        data,
        required={
            "joints",
            "gripper",
            "leader",
            "leader_motion",
            "motion",
        },
        label="reset pose",
    )
    system = teleop.system

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

    gripper = _load_a1_gripper_home(required_table(data, "gripper"))
    if not (
        system.gripper.stroke_min_mm
        <= gripper.closed_stroke_mm
        <= system.gripper.stroke_max_mm
    ):
        raise ValueError("reset gripper target is outside the system stroke range")

    leader = _load_leader_home(
        required_table(data, "leader"),
        teleop.leader,
        dof=teleop.bridge.dof,
        gripper_key=teleop.gripper.source_key,
    )
    leader_motion = _load_leader_motion(required_table(data, "leader_motion"))
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
    topics = system.topics
    home = HomePose(
        path=path,
        names=names,
        positions=positions,
        topics=HomeTopics(
            joint_states=topics.joint_states,
            target=topics.joint_target,
            staged_command=topics.staged_command,
            relay_enable=topics.motion_enable,
            relay_status=topics.relay_status,
            gripper_target=topics.gripper_target,
        ),
        motion=HomeMotion(
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
        ),
        gripper=gripper,
        leader=leader,
        leader_motion=leader_motion,
    )
    validate_home_pose(home)
    return home


def validate_home_pose(home: HomePose) -> None:
    for field, value in home.motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"motion.{field} must be positive")
    if home.gripper.publish_hz <= 0 or home.gripper.publish_s <= 0:
        raise ValueError("gripper publish_hz/publish_s must be positive")
    for field, value in home.leader_motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"leader_motion.{field} must be positive")


def _load_a1_gripper_home(data: dict) -> A1GripperHome:
    require_exact_keys(
        data,
        required={"enabled", "closed_stroke_mm", "publish_hz", "publish_s"},
        label="reset gripper",
    )
    return A1GripperHome(
        enabled=boolean(data, "enabled"),
        closed_stroke_mm=floating(data, "closed_stroke_mm"),
        publish_hz=floating(data, "publish_hz"),
        publish_s=floating(data, "publish_s"),
    )


def _load_leader_home(
    data: dict,
    config: TeleopLeaderConfig,
    *,
    dof: int,
    gripper_key: str,
) -> LeaderHome:
    require_exact_keys(
        data,
        required={"enabled", "action_keys", "action_values"},
        label="reset leader",
    )
    expected_keys = (*(f"joint{index}.pos" for index in range(dof)), gripper_key)
    keys = string_tuple(data, "action_keys", len(expected_keys))
    if keys != expected_keys:
        raise ValueError(
            f"reset leader action keys must be {list(expected_keys)}, got {list(keys)}"
        )
    values = float_tuple(data, "action_values", len(keys))
    return LeaderHome(
        enabled=boolean(data, "enabled"),
        config=config,
        action=dict(zip(keys, values, strict=True)),
    )


def _load_leader_motion(data: dict) -> LeaderMotion:
    require_exact_keys(
        data,
        required={
            "hz",
            "min_duration_s",
            "max_velocity_units_s",
            "hold_s",
            "goal_tolerance_units",
            "gripper_goal_tolerance_units",
        },
        label="reset leader_motion",
    )
    return LeaderMotion(
        hz=floating(data, "hz"),
        min_duration_s=floating(data, "min_duration_s"),
        max_velocity_units_s=floating(data, "max_velocity_units_s"),
        hold_s=floating(data, "hold_s"),
        goal_tolerance_units=floating(data, "goal_tolerance_units"),
        gripper_goal_tolerance_units=floating(data, "gripper_goal_tolerance_units"),
    )


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
