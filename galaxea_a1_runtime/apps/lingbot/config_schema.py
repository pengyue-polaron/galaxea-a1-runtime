"""Typed LingBot deployment configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from galaxea_a1_runtime.configuration.system import SystemConfig


OrientationMode = Literal["hold-current", "model-quat"]
PoseMode = Literal["absolute", "episode-relative"]
TextEncoderDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class LingBotSessionConfig:
    tmux: str


@dataclass(frozen=True)
class LingBotServerConfig:
    host: str
    port: int
    connect_timeout_s: float
    close_timeout_s: float
    prompt: str


@dataclass(frozen=True)
class LingBotPolicyServerConfig:
    tmux: str
    checkout: Path
    python: Path
    base_model: Path
    checkpoint: Path
    model_root: Path
    save_root: Path
    master_port: int
    startup_timeout_s: float
    expected_weight_size_bytes: int
    text_encoder_device: TextEncoderDevice
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
    no_kv_update: bool
    max_model_calls: int
    execute_frames: int
    kv_observations_per_frame: int
    condition_on_ee_state: bool
    initial_ee_pose: tuple[float, ...] | None
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
class LingBotServoConfig:
    gain: float
    max_extra_m: float
    settle_s: float
    tolerance_m: float
    corrections: int
    cache_actual_feedback: bool


@dataclass(frozen=True)
class LingBotConfig:
    path: Path
    system: SystemConfig
    session: LingBotSessionConfig
    server: LingBotServerConfig
    policy_server: LingBotPolicyServerConfig
    execution: LingBotExecutionConfig
    observations: LingBotObservationConfig
    action: LingBotActionModeConfig
    servo: LingBotServoConfig
