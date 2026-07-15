"""Typed teleoperation configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.teleop.joint_mapping import JointMappingConfig


@dataclass(frozen=True)
class TeleopRuntimeConfig:
    prefix: str
    run_dir: str


@dataclass(frozen=True)
class TeleopLeaderConfig:
    port: str
    id: str
    use_degrees: bool


@dataclass(frozen=True)
class TeleopBridgeConfig:
    hz: float
    dof: int
    mapping: JointMappingConfig
    a1_state_timeout_s: float


@dataclass(frozen=True)
class TeleopGripperConfig:
    enabled: bool
    source_key: str
    invert: bool


@dataclass(frozen=True)
class TeleopCollectionConfig:
    data_root: Path
    state_mode: StateMode
    fps: float
    max_duration_s: float
    auto_reset_after_save: bool
    jpeg_quality: int
    ready_timeout_s: float
    max_joint_action_step_rad: float


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    system: SystemConfig
    runtime: TeleopRuntimeConfig
    leader: TeleopLeaderConfig
    bridge: TeleopBridgeConfig
    gripper: TeleopGripperConfig
    collection: TeleopCollectionConfig
