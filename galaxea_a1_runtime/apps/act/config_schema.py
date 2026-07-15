"""Typed ACT deployment configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.system import SystemConfig


@dataclass(frozen=True)
class ActSessionConfig:
    tmux: str


@dataclass(frozen=True)
class ActRuntimeConfig:
    prefix: str


@dataclass(frozen=True)
class ActPolicyConfig:
    checkpoint: Path
    device: str
    disable_backbone_download: bool
    deployment_ready: bool


@dataclass(frozen=True)
class ActExecutionConfig:
    execute: bool
    step_mode: bool
    execute_steps_per_inference: int
    control_hz: float
    max_model_calls: int
    print_actions: bool
    preview_steps: int


@dataclass(frozen=True)
class ActConfig:
    path: Path
    system: SystemConfig
    session: ActSessionConfig
    runtime: ActRuntimeConfig
    policy: ActPolicyConfig
    execution: ActExecutionConfig
