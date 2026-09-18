"""Typed TFP-Ultra deployment schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embodied_ops import TaskCatalog

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.models.backend import CodeBackendConfig


@dataclass(frozen=True)
class TFPEngineConfig:
    device: str
    seed: int
    package_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TFPModelConfig:
    model_id: str
    checkpoint: Path
    checkpoint_format: str
    checkpoint_step: int
    checkpoint_size: int
    checkpoint_sha256: str
    metadata: Path
    metadata_info_sha256: str
    metadata_stats_sha256: str
    dataset_repo_id: str
    dataset_revision: str
    action_names: tuple[str, ...]


@dataclass(frozen=True)
class TFPObservationConfig:
    front_key: str
    wrist_key: str


@dataclass(frozen=True)
class TFPResetConfig:
    before_run: bool
    config: Path


@dataclass(frozen=True)
class TFPExecutionConfig:
    execute: bool
    max_actions: int
    exec_rate: float
    observation_timeout_s: float
    print_actions: bool


@dataclass(frozen=True)
class TFPDiagnosticsConfig:
    enabled: bool
    output_dir: Path
    print_every_queries: int
    include_belief_vectors: bool


@dataclass(frozen=True)
class TFPConfig:
    path: Path
    repo_root: Path
    deployment_id: str
    deployment_ready: bool
    task_id: str
    system: SystemConfig
    backend: CodeBackendConfig
    engine: TFPEngineConfig
    task_catalog: TaskCatalog
    model: TFPModelConfig
    observations: TFPObservationConfig
    reset: TFPResetConfig
    execution: TFPExecutionConfig
    diagnostics: TFPDiagnosticsConfig
