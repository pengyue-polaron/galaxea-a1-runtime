"""Typed configuration objects for the Galaxea A1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .constants import (
    DEFAULT_MAX_COMMAND_AGE_S,
    EE_TARGET_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    RELAY_ENABLE_TOPIC,
    RELAY_STATUS_TOPIC,
    STAGED_ARM_COMMAND_TOPIC,
)
from .schema import ActionMode


class RuntimeProfile(StrEnum):
    STATIC = "static"
    SAFE = "safe"
    COLLECT = "collect"
    INFER = "infer"
    DIRECT_DEBUG = "direct-debug"


@dataclass(frozen=True)
class SafetyConfig:
    max_command_age_s: float = DEFAULT_MAX_COMMAND_AGE_S
    max_eef_delta_m: float | None = None
    max_rot_delta_rad: float | None = None


@dataclass(frozen=True)
class TopicConfig:
    ee_target: str = EE_TARGET_TOPIC
    staged_arm_command: str = STAGED_ARM_COMMAND_TOPIC
    relay_enable: str = RELAY_ENABLE_TOPIC
    relay_status: str = RELAY_STATUS_TOPIC
    gripper_command: str = GRIPPER_COMMAND_TOPIC


@dataclass(frozen=True)
class RuntimeConfig:
    profile: RuntimeProfile = RuntimeProfile.STATIC
    action_mode: ActionMode = ActionMode.EEF_DELTA
    topics: TopicConfig = TopicConfig()
    safety: SafetyConfig = SafetyConfig()
    serial: str = "/dev/a1"

    @property
    def touches_hardware(self) -> bool:
        return self.profile != RuntimeProfile.STATIC


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    root: Path
    fps: int
    robot_type: str = "galaxea_a1"
    use_videos: bool = True
    vcodec: str = "libsvtav1"

    def validate(self) -> None:
        if "/" not in self.repo_id:
            raise ValueError("repo_id should be namespaced, for example 'user/a1_task'")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
