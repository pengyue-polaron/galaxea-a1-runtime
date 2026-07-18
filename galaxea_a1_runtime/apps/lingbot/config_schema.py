"""Typed LingBot deployment configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.configuration.tasks import TaskCatalog
from galaxea_a1_runtime.models.backend import CodeBackendConfig
from galaxea_a1_runtime.models.config import ModelArtifactConfig


PoseMode = Literal["absolute", "episode-relative"]
AttentionMode = Literal["torch", "flashattn"]
TextEncoderDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class LingBotServerConfig:
    host: str
    port: int
    connect_timeout_s: float
    close_timeout_s: float


@dataclass(frozen=True)
class LingBotPolicyServerConfig:
    backend: CodeBackendConfig
    model: ModelArtifactConfig
    vendor_config: str
    save_root: Path
    master_port: int
    world_size: int
    startup_timeout_s: float
    shutdown_timeout_s: float
    expected_weight_sha256: str
    expected_transformer_config_sha256: str
    model_action_dim: int
    action_channel_ids: tuple[int, ...]
    text_encoder_device: TextEncoderDevice
    enable_offload: bool
    attention_mode: AttentionMode
    seed: int
    height: int
    width: int
    frame_chunk_size: int
    action_per_frame: int
    attention_window: int
    guidance_scale: float
    action_guidance_scale: float
    video_inference_steps: int
    action_inference_steps: int
    snr_shift: float
    action_snr_shift: float
    q01_source: tuple[float, ...]
    q99_source: tuple[float, ...]
    deployment_ready: bool


@dataclass(frozen=True)
class LingBotExecutionConfig:
    execute: bool
    step_mode: bool
    step_actions: bool
    max_model_calls: int
    execute_frames: int
    kv_observations_per_frame: int
    exec_rate: float
    print_actions: bool
    review_deadband_m: float


@dataclass(frozen=True)
class LingBotObservationConfig:
    front_key: str
    wrist_key: str


@dataclass(frozen=True)
class LingBotActionModeConfig:
    pose_mode: PoseMode


@dataclass(frozen=True)
class LingBotRecordingConfig:
    agent_view_enabled: bool
    output_root: Path


@dataclass(frozen=True)
class LingBotConfig:
    path: Path
    system: SystemConfig
    server: LingBotServerConfig
    task_catalog: TaskCatalog
    policy_server: LingBotPolicyServerConfig
    execution: LingBotExecutionConfig
    observations: LingBotObservationConfig
    action: LingBotActionModeConfig
    recording: LingBotRecordingConfig
