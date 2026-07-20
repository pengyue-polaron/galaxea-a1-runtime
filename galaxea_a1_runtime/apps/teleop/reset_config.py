"""Strict tracked dual-device reset configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot_robot_galaxea_a1 import (
    DEFAULT_GRIPPER_INPUT_KEY,
    DEFAULT_LEADER_JOINT_KEYS,
)

from galaxea_a1_runtime.apps.reset.config import A1HomePose, load_a1_home_pose
from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple,
    floating,
    load_toml,
    repo_path,
    require_exact_keys,
    required_table,
    string,
    string_tuple,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig, TeleopLeaderConfig


ROOT_DIR = Path(__file__).resolve().parents[3]


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
    a1: A1HomePose
    leader: LeaderHome
    leader_motion: LeaderMotion


def load_home_pose(path: Path, *, teleop: TeleopConfig) -> HomePose:
    path, repo_root, data = load_toml(path, repo_root=ROOT_DIR)
    require_exact_keys(
        data,
        required={"a1", "leader", "leader_motion"},
        label="reset pose",
    )
    a1_ref = required_table(data, "a1")
    require_exact_keys(a1_ref, required={"config"}, label="A1 reset reference")
    a1 = load_a1_home_pose(
        repo_path(repo_root, string(a1_ref, "config")),
        system=teleop.system,
        repo_root=repo_root,
    )

    leader = _load_leader_home(
        required_table(data, "leader"),
        teleop.leader,
    )
    leader_motion = _load_leader_motion(required_table(data, "leader_motion"))
    home = HomePose(
        path=path,
        a1=a1,
        leader=leader,
        leader_motion=leader_motion,
    )
    validate_home_pose(home)
    return home


def validate_home_pose(home: HomePose) -> None:
    for field, value in home.leader_motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"leader_motion.{field} must be positive")


def _load_leader_home(
    data: dict,
    config: TeleopLeaderConfig,
) -> LeaderHome:
    require_exact_keys(
        data,
        required={"enabled", "action_keys", "action_values"},
        label="reset leader",
    )
    expected_keys = (*DEFAULT_LEADER_JOINT_KEYS, DEFAULT_GRIPPER_INPUT_KEY)
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
