"""Typed teleoperation configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.system import SystemConfig


@dataclass(frozen=True)
class JointMappingConfig:
    scale: tuple[float, ...]
    sign: tuple[float, ...]
    bias_rad: tuple[float, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]

    def validate(self, dof: int) -> None:
        for name, values in (
            ("scale", self.scale),
            ("sign", self.sign),
            ("bias_rad", self.bias_rad),
            ("lower_limits", self.lower_limits),
            ("upper_limits", self.upper_limits),
        ):
            if len(values) != dof:
                raise ValueError(f"{name} expects {dof} values, got {len(values)}")
        for lo, hi in zip(self.lower_limits, self.upper_limits, strict=True):
            if lo > hi:
                raise ValueError(f"invalid joint limit: lower={lo} upper={hi}")


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
    motor_write_retries: int


@dataclass(frozen=True)
class TeleopBridgeConfig:
    hz: float
    mapping: JointMappingConfig


@dataclass(frozen=True)
class TeleopGripperConfig:
    source_min: float
    source_max: float
    invert: bool
    saturate_out_of_range: bool


@dataclass(frozen=True)
class TeleopCollectionConfig:
    dataset_root: Path
    repo_id_prefix: str
    fps: float
    max_duration_s: float
    auto_reset_after_save: bool
    auto_reset_after_discard: bool
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
