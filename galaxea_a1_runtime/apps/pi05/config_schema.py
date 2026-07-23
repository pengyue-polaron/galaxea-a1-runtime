"""Typed OpenPI pi0.5 deployment schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from galaxea_a1_runtime.configuration.system import SystemConfig
from embodied_ops import TaskCatalog
from galaxea_a1_runtime.models.backend import CodeBackendConfig
from galaxea_a1_runtime.models.config import ModelArtifactConfig


PoseMode = Literal["absolute", "episode-relative"]


@dataclass(frozen=True)
class Pi05EngineConfig:
    jax_platform: Literal["cuda"]
    xla_memory_fraction: float
    seed: int
    sampling_steps: int


@dataclass(frozen=True)
class Pi05ModelContract:
    train_config: str
    checkpoint_format: str
    parameter_set: str
    norm_stats_path: Path
    pose_mode: PoseMode
    action_horizon: int
    state_dim: int
    source_action_dim: int
    model_action_dim: int


@dataclass(frozen=True)
class Pi05SessionConfig:
    tmux: str
    model_tmux: str
    startup_timeout_s: float


@dataclass(frozen=True)
class Pi05ServerConfig:
    host: str
    port: int
    connect_timeout_s: float
    close_timeout_s: float


@dataclass(frozen=True)
class Pi05ObservationConfig:
    front_key: str
    wrist_key: str


@dataclass(frozen=True)
class Pi05ExecutionConfig:
    execute: bool
    step_mode: bool
    step_actions: bool
    max_model_calls: int
    execute_actions_per_inference: int
    exec_rate: float
    print_actions: bool
    review_deadband_m: float


@dataclass(frozen=True)
class Pi05Config:
    path: Path
    deployment_id: str
    deployment_ready: bool
    system: SystemConfig
    backend: CodeBackendConfig
    engine: Pi05EngineConfig
    model: ModelArtifactConfig
    model_contract: Pi05ModelContract
    task_catalog: TaskCatalog
    session: Pi05SessionConfig
    server: Pi05ServerConfig
    observations: Pi05ObservationConfig
    execution: Pi05ExecutionConfig
