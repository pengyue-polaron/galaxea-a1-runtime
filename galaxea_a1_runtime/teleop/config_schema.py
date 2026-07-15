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
    bridge_startup_timeout_s: float
    bridge_stop_timeout_s: float


@dataclass(frozen=True)
class TeleopResetConfig:
    config: Path


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
    source_min: float
    source_max: float
    invert: bool
    saturate_out_of_range: bool


@dataclass(frozen=True)
class TeleopCollectionConfig:
    data_root: Path
    state_mode: StateMode
    fps: float
    max_duration_s: float
    auto_reset_after_save: bool
    auto_reset_after_discard: bool
    jpeg_quality: int
    ready_timeout_s: float
    max_joint_action_step_rad: float


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    system: SystemConfig
    runtime: TeleopRuntimeConfig
    reset: TeleopResetConfig
    leader: TeleopLeaderConfig
    bridge: TeleopBridgeConfig
    gripper: TeleopGripperConfig
    collection: TeleopCollectionConfig
