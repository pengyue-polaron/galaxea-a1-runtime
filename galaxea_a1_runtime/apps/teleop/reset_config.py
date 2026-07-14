"""Tracked dual-device reset configuration."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.system import load_system_config


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
    port: str
    id: str
    use_degrees: bool
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
    leader: LeaderHome | None
    leader_motion: LeaderMotion | None


def load_home_pose(path: Path) -> HomePose:
    path = path.expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    data = tomllib.loads(path.read_text())
    names = tuple(_required_list(data["joints"], "names", str))
    positions = tuple(
        float(value) for value in _required_list(data["joints"], "position_rad", float)
    )
    if len(names) != len(positions):
        raise ValueError(
            "joints.names and joints.position_rad must have the same length"
        )
    system_path = ROOT_DIR / str(data["system"]["config"])
    system = load_system_config(system_path, repo_root=ROOT_DIR)
    if names != system.joint_safety.names:
        raise ValueError("reset joint names must match configs/system/a1.toml")
    a1_gripper = _load_a1_gripper_home(data)
    if (
        not system.gripper.stroke_min_mm
        <= a1_gripper.closed_stroke_mm
        <= system.gripper.stroke_max_mm
    ):
        raise ValueError("reset gripper target is outside configs/system/a1.toml range")
    motion = data["motion"]
    leader = _load_leader_home(data)
    home = HomePose(
        path=path,
        names=names,
        positions=positions,
        topics=HomeTopics(
            joint_states=system.topics.joint_states,
            target=system.topics.joint_target,
            staged_command=system.topics.staged_command,
            relay_enable=system.topics.motion_enable,
            relay_status=system.topics.relay_status,
            gripper_target=system.topics.gripper_target,
        ),
        motion=HomeMotion(
            hz=float(motion["hz"]),
            min_duration_s=float(motion["min_duration_s"]),
            max_velocity_rad_s=float(motion["max_velocity_rad_s"]),
            tracker_alignment_timeout_s=float(motion["tracker_alignment_timeout_s"]),
            tracker_alignment_tolerance_rad=system.joint_safety.initial_alignment_tolerance_rad,
            relay_enable_timeout_s=system.relay.enable_timeout_s,
            max_relay_status_age_s=system.relay.max_status_age_s,
            hold_s=float(motion["hold_s"]),
            goal_tolerance_rad=float(motion["goal_tolerance_rad"]),
        ),
        gripper=a1_gripper,
        leader=leader[0],
        leader_motion=leader[1],
    )
    validate_home_pose(home)
    return home


def validate_home_pose(home: HomePose) -> None:
    if not home.names:
        raise ValueError("home pose must contain at least one joint")
    if any(not name for name in home.names):
        raise ValueError("joint names must be non-empty")
    if any(not math.isfinite(value) for value in home.positions):
        raise ValueError("joint positions must be finite")
    for field, value in home.motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"motion.{field} must be positive")
    if home.gripper.closed_stroke_mm < 0:
        raise ValueError("gripper.closed_stroke_mm must be non-negative")
    if home.gripper.publish_hz <= 0 or home.gripper.publish_s <= 0:
        raise ValueError("gripper publish_hz/publish_s must be positive")
    if home.leader is not None:
        if not home.leader.port:
            raise ValueError("leader.port must be non-empty")
        if not home.leader.action:
            raise ValueError("leader.action must contain at least one target")
    if home.leader_motion is not None:
        for field, value in home.leader_motion.__dict__.items():
            if value <= 0:
                raise ValueError(f"leader_motion.{field} must be positive")


def _load_leader_home(
    data: dict[str, Any],
) -> tuple[LeaderHome | None, LeaderMotion | None]:
    leader_data = data.get("leader")
    if leader_data is None:
        return None, None
    if not isinstance(leader_data, dict):
        raise ValueError("leader must be a table")
    enabled = bool(leader_data.get("enabled", True))
    keys = _required_list(leader_data, "action_keys", str)
    values = _required_number_list(leader_data, "action_values")
    if len(keys) != len(values):
        raise ValueError(
            "leader.action_keys and leader.action_values must have the same length"
        )

    motion_data = data.get("leader_motion")
    if not isinstance(motion_data, dict):
        raise ValueError("leader_motion table is required when leader is configured")
    return (
        LeaderHome(
            enabled=enabled,
            port=_required_string(leader_data, "port"),
            id=_required_string(leader_data, "id"),
            use_degrees=bool(leader_data.get("use_degrees", True)),
            action=dict(zip(keys, (float(value) for value in values), strict=True)),
        ),
        LeaderMotion(
            hz=float(motion_data["hz"]),
            min_duration_s=float(motion_data["min_duration_s"]),
            max_velocity_units_s=float(motion_data["max_velocity_units_s"]),
            hold_s=float(motion_data["hold_s"]),
            goal_tolerance_units=float(motion_data["goal_tolerance_units"]),
            gripper_goal_tolerance_units=float(
                motion_data.get(
                    "gripper_goal_tolerance_units",
                    motion_data["goal_tolerance_units"],
                )
            ),
        ),
    )


def _load_a1_gripper_home(data: dict[str, Any]) -> A1GripperHome:
    gripper = data.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("gripper table is required")
    return A1GripperHome(
        enabled=bool(gripper.get("enabled", True)),
        closed_stroke_mm=float(gripper["closed_stroke_mm"]),
        publish_hz=float(gripper["publish_hz"]),
        publish_s=float(gripper["publish_s"]),
    )


def _required_list(data: dict[str, Any], key: str, item_type: type) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, item_type) for item in value
    ):
        raise ValueError(f"{key} must be a list of {item_type.__name__}")
    return value


def _required_number_list(data: dict[str, Any], key: str) -> list[float | int]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int | float) for item in value
    ):
        raise ValueError(f"{key} must be a number list")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
