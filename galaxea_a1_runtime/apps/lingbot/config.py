"""Composed LingBot backend, model, deployment, and System configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.lingbot.config_runtime import bash_config
from galaxea_a1_runtime.apps.lingbot.config_schema import (
    AttentionMode,
    LingBotActionModeConfig,
    LingBotConfig,
    LingBotExecutionConfig,
    LingBotObservationConfig,
    LingBotPolicyServerConfig,
    LingBotRecordingConfig,
    LingBotServerConfig,
    PoseMode,
    TextEncoderDevice,
)
from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple,
    floating,
    identifier,
    integer,
    integer_tuple,
    load_toml,
    referenced_config,
    require_exact_keys,
    repo_path,
    required_table,
    string,
)
from galaxea_a1_runtime.configuration.paths import LINGBOT_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.configuration.tasks import load_task_catalog
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.models.backend import CodeBackendConfig, parse_code_backend
from galaxea_a1_runtime.models.config import ModelArtifactConfig, load_model_config
from galaxea_a1_runtime.models.registry import resolve_registered_model
from galaxea_a1_runtime.schema import LINGBOT_EEF_ACTION_CHANNEL_IDS


__all__ = ["bash_config", "load_lingbot_config"]


@dataclass(frozen=True)
class _EngineConfig:
    text_encoder_device: TextEncoderDevice
    enable_offload: bool
    attention_mode: AttentionMode
    seed: int
    height: int
    width: int
    attention_window: int
    guidance_scale: float
    action_guidance_scale: float
    video_inference_steps: int
    action_inference_steps: int
    snr_shift: float
    action_snr_shift: float
    world_size: int


@dataclass(frozen=True)
class _ModelContract:
    vendor_config: str
    pose_mode: PoseMode
    frame_chunk_size: int
    action_per_frame: int
    model_action_dim: int
    action_channel_ids: tuple[int, ...]
    q01_source: tuple[float, ...]
    q99_source: tuple[float, ...]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / LINGBOT_CONFIG


def load_lingbot_config(
    path: Path,
    *,
    repo_root: Path | None = None,
    model_selector: str | None = None,
) -> LingBotConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "system",
            "backend",
            "model",
            "tasks",
            "deployment",
            "session",
            "server",
            "observations",
            "execution",
            "recording",
        },
        label="LingBot deployment config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    backend, engine = _load_backend(
        referenced_config(data, repo_root, key="backend"), repo_root
    )
    default_model = load_model_config(
        referenced_config(data, repo_root, key="model"), repo_root=repo_root
    )
    model = (
        default_model
        if model_selector is None
        else resolve_registered_model(
            model_selector,
            repo_root=repo_root,
            backend=backend.backend_id,
        )
    )
    if backend.adapter != "lingbot_va" or model.backend != backend.backend_id:
        raise ValueError(
            "LingBot deployment backend/model mismatch: "
            f"adapter={backend.adapter!r}, backend={backend.backend_id!r}, "
            f"model.backend={model.backend!r}"
        )
    if model.artifact_format != "diffusers":
        raise ValueError("LingBot model artifact_format must be 'diffusers'")
    contract = _load_model_contract(model)
    task_catalog = load_task_catalog(
        referenced_config(data, repo_root, key="tasks"), repo_root=repo_root
    )

    deployment = required_table(data, "deployment")
    session = required_table(data, "session")
    server = required_table(data, "server")
    observations = required_table(data, "observations")
    execution = required_table(data, "execution")
    recording = required_table(data, "recording")
    require_exact_keys(deployment, required={"id", "ready"}, label="LingBot deployment")
    require_exact_keys(
        session,
        required={
            "master_port",
            "startup_timeout_s",
            "shutdown_timeout_s",
        },
        label="session",
    )
    require_exact_keys(
        server,
        required={"host", "port", "connect_timeout_s", "close_timeout_s"},
        label="server",
    )
    require_exact_keys(
        observations, required={"front_key", "wrist_key"}, label="observations"
    )
    require_exact_keys(
        execution,
        required={
            "execute",
            "step_mode",
            "step_actions",
            "max_model_calls",
            "execute_frames",
            "kv_observations_per_frame",
            "exec_rate",
            "print_actions",
            "review_deadband_m",
        },
        label="execution",
    )
    require_exact_keys(
        recording,
        required={"agent_view_enabled", "output_root"},
        label="recording",
    )

    deployment_id = identifier(string(deployment, "id"), label="deployment.id")
    transformer_weight = _manifest_file(
        model, "transformer/diffusion_pytorch_model.safetensors"
    )
    transformer_config = _manifest_file(model, "transformer/config.json")
    recording_output_root = repo_path(repo_root, string(recording, "output_root"))
    if not recording_output_root.is_relative_to((repo_root / "outputs").resolve()):
        raise ValueError("recording.output_root must remain under outputs/")
    deployment_ready = boolean(deployment, "ready")
    config = LingBotConfig(
        path=path,
        system=system,
        server=LingBotServerConfig(
            host=string(server, "host"),
            port=integer(server, "port"),
            connect_timeout_s=floating(server, "connect_timeout_s"),
            close_timeout_s=floating(server, "close_timeout_s"),
        ),
        task_catalog=task_catalog,
        policy_server=LingBotPolicyServerConfig(
            backend=backend,
            model=model,
            vendor_config=contract.vendor_config,
            save_root=Path(
                os.path.abspath(repo_root / "outputs" / "inference" / deployment_id)
            ),
            master_port=integer(session, "master_port"),
            world_size=engine.world_size,
            startup_timeout_s=floating(session, "startup_timeout_s"),
            shutdown_timeout_s=floating(session, "shutdown_timeout_s"),
            expected_weight_sha256=transformer_weight.sha256,
            expected_transformer_config_sha256=transformer_config.sha256,
            model_action_dim=contract.model_action_dim,
            action_channel_ids=contract.action_channel_ids,
            text_encoder_device=engine.text_encoder_device,
            enable_offload=engine.enable_offload,
            attention_mode=engine.attention_mode,
            seed=engine.seed,
            height=engine.height,
            width=engine.width,
            frame_chunk_size=contract.frame_chunk_size,
            action_per_frame=contract.action_per_frame,
            attention_window=engine.attention_window,
            guidance_scale=engine.guidance_scale,
            action_guidance_scale=engine.action_guidance_scale,
            video_inference_steps=engine.video_inference_steps,
            action_inference_steps=engine.action_inference_steps,
            snr_shift=engine.snr_shift,
            action_snr_shift=engine.action_snr_shift,
            q01_source=contract.q01_source,
            q99_source=contract.q99_source,
            deployment_ready=deployment_ready,
        ),
        execution=LingBotExecutionConfig(
            execute=boolean(execution, "execute"),
            step_mode=boolean(execution, "step_mode"),
            step_actions=boolean(execution, "step_actions"),
            max_model_calls=integer(execution, "max_model_calls"),
            execute_frames=integer(execution, "execute_frames"),
            kv_observations_per_frame=integer(execution, "kv_observations_per_frame"),
            exec_rate=floating(execution, "exec_rate"),
            print_actions=boolean(execution, "print_actions"),
            review_deadband_m=floating(execution, "review_deadband_m"),
        ),
        observations=LingBotObservationConfig(
            front_key=string(observations, "front_key"),
            wrist_key=string(observations, "wrist_key"),
        ),
        action=LingBotActionModeConfig(
            pose_mode=contract.pose_mode,
        ),
        recording=LingBotRecordingConfig(
            agent_view_enabled=boolean(recording, "agent_view_enabled"),
            output_root=recording_output_root,
        ),
    )
    validate_lingbot_config(config)
    return config


def _load_backend(
    path: Path, repo_root: Path
) -> tuple[CodeBackendConfig, _EngineConfig]:
    _, _, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"backend", "source", "environment", "engine"},
        label="LingBot backend config",
    )
    backend = parse_code_backend(
        backend=required_table(data, "backend"),
        source=required_table(data, "source"),
        environment=required_table(data, "environment"),
        repo_root=repo_root,
    )
    return backend, _parse_engine(required_table(data, "engine"))


def _parse_engine(engine_data: dict[str, Any]) -> _EngineConfig:
    require_exact_keys(
        engine_data,
        required={
            "text_encoder_device",
            "enable_offload",
            "attention_mode",
            "seed",
            "height",
            "width",
            "attention_window",
            "guidance_scale",
            "action_guidance_scale",
            "video_inference_steps",
            "action_inference_steps",
            "snr_shift",
            "action_snr_shift",
            "world_size",
        },
        label="LingBot backend engine",
    )
    return _EngineConfig(
        text_encoder_device=_text_encoder_device(
            string(engine_data, "text_encoder_device")
        ),
        enable_offload=boolean(engine_data, "enable_offload"),
        attention_mode=_attention_mode(string(engine_data, "attention_mode")),
        seed=integer(engine_data, "seed"),
        height=integer(engine_data, "height"),
        width=integer(engine_data, "width"),
        attention_window=integer(engine_data, "attention_window"),
        guidance_scale=floating(engine_data, "guidance_scale"),
        action_guidance_scale=floating(engine_data, "action_guidance_scale"),
        video_inference_steps=integer(engine_data, "video_inference_steps"),
        action_inference_steps=integer(engine_data, "action_inference_steps"),
        snr_shift=floating(engine_data, "snr_shift"),
        action_snr_shift=floating(engine_data, "action_snr_shift"),
        world_size=integer(engine_data, "world_size"),
    )


def _load_model_contract(model: ModelArtifactConfig) -> _ModelContract:
    _, _, data = load_toml(model.contract, repo_root=model.repo_root)
    require_exact_keys(
        data, required={"lingbot", "normalization"}, label="LingBot model contract"
    )
    lingbot = required_table(data, "lingbot")
    normalization = required_table(data, "normalization")
    require_exact_keys(
        lingbot,
        required={
            "vendor_config",
            "pose_mode",
            "frame_chunk_size",
            "action_per_frame",
            "model_action_dim",
            "action_channel_ids",
        },
        label="LingBot model contract",
    )
    require_exact_keys(
        normalization,
        required={"method", "q01_source", "q99_source"},
        label="LingBot normalization contract",
    )
    if string(normalization, "method") != "quantiles":
        raise ValueError("LingBot normalization.method must be 'quantiles'")
    return _ModelContract(
        vendor_config=string(lingbot, "vendor_config"),
        pose_mode=_pose_mode(string(lingbot, "pose_mode")),
        frame_chunk_size=integer(lingbot, "frame_chunk_size"),
        action_per_frame=integer(lingbot, "action_per_frame"),
        model_action_dim=integer(lingbot, "model_action_dim"),
        action_channel_ids=integer_tuple(lingbot, "action_channel_ids", min_len=1),
        q01_source=float_tuple(normalization, "q01_source"),
        q99_source=float_tuple(normalization, "q99_source"),
    )


def validate_lingbot_config(config: LingBotConfig) -> None:
    if not 1 <= config.server.port <= 65535:
        raise ValueError("server.port must be in [1, 65535]")
    if min(config.server.connect_timeout_s, config.server.close_timeout_s) <= 0:
        raise ValueError("server connection timeouts must be positive")
    policy = config.policy_server
    if (
        not 1 <= policy.master_port <= 65535
        or min(policy.startup_timeout_s, policy.shutdown_timeout_s) <= 0
    ):
        raise ValueError(
            "model master_port must be in [1, 65535] and process timeouts must be positive"
        )
    if policy.world_size != 1:
        raise ValueError("LingBot backend world_size must be 1")
    if (
        min(
            policy.height,
            policy.width,
            policy.frame_chunk_size,
            policy.action_per_frame,
        )
        <= 0
    ):
        raise ValueError("LingBot backend/model dimensions must be positive")
    if policy.frame_chunk_size < 2:
        raise ValueError(
            "LingBot frame_chunk_size must include the conditioned first frame and a predicted frame"
        )
    if (
        min(
            policy.attention_window,
            policy.video_inference_steps,
            policy.action_inference_steps,
        )
        <= 0
    ):
        raise ValueError("LingBot inference settings must be positive")
    if policy.model_action_dim <= max(policy.action_channel_ids, default=-1):
        raise ValueError("LingBot action_channel_ids exceed model_action_dim")
    if policy.action_channel_ids != LINGBOT_EEF_ACTION_CHANNEL_IDS:
        raise ValueError(
            "LingBot checkpoint action_channel_ids do not match the shared A1 EEF schema"
        )
    if policy.deployment_ready and (not policy.q01_source or not policy.q99_source):
        raise ValueError("deployment-ready LingBot config requires q01/q99 statistics")
    if len(policy.action_channel_ids) != len(policy.q01_source) or len(
        policy.q01_source
    ) != len(policy.q99_source):
        raise ValueError(
            "LingBot action channels and quantiles must have equal lengths"
        )
    if any(
        lo >= hi for lo, hi in zip(policy.q01_source, policy.q99_source, strict=True)
    ):
        raise ValueError("LingBot q01 values must be lower than q99 values")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if (
        min(
            config.execution.execute_frames,
            config.execution.kv_observations_per_frame,
        )
        <= 0
    ):
        raise ValueError("LingBot execution frame counts must be positive")
    if config.execution.execute_frames > policy.frame_chunk_size:
        raise ValueError(
            "execution.execute_frames cannot exceed the model frame_chunk_size"
        )
    if policy.action_per_frame % config.execution.kv_observations_per_frame:
        raise ValueError(
            "LingBot action_per_frame must be divisible by kv_observations_per_frame"
        )
    if config.execution.exec_rate <= 0 or config.execution.review_deadband_m < 0:
        raise ValueError(
            "LingBot execution rate must be positive and deadband non-negative"
        )
    if config.execution.execute and not policy.deployment_ready:
        raise ValueError("execution.execute requires deployment.ready=true")
    if config.system.cameras.front.backend != "realsense":
        raise ValueError("LingBot front camera must use the RealSense backend")


def _manifest_file(model: ModelArtifactConfig, path: str):
    for item in model.manifest.files:
        if item.path.as_posix() == path:
            return item
    raise ValueError(f"LingBot model manifest is missing required path: {path}")


def _pose_mode(value: str) -> PoseMode:
    if value not in ("absolute", "episode-relative"):
        raise ValueError(f"unsupported LingBot model pose_mode: {value!r}")
    return value


def _text_encoder_device(value: str) -> TextEncoderDevice:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"unsupported engine.text_encoder_device: {value!r}")
    return value


def _attention_mode(value: str) -> AttentionMode:
    if value not in ("torch", "flashattn"):
        raise ValueError(f"unsupported engine.attention_mode: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Read the composed A1 LingBot deployment config."
    )
    parser.add_argument("config", nargs="?", type=Path, default=LINGBOT_CONFIG)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used to resolve tracked relative paths",
    )
    parser.add_argument(
        "--model",
        help="registered model id, pinned id, or unique descriptor name",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit validated shell assignments for a runtime supervisor",
    )
    args = parser.parse_args(argv)
    config = load_lingbot_config(
        args.config,
        repo_root=args.repo_root,
        model_selector=args.model,
    )
    print(bash_config(config) if args.shell else config.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
