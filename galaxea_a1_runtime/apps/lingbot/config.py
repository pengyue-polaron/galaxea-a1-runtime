"""Git-tracked LingBot-VA runtime configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple,
    floating,
    integer,
    load_toml,
    referenced_config,
    require_exact_keys,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
    text as _text,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.configuration.cli import run_config_renderer
from galaxea_a1_runtime.configuration.paths import LINGBOT_CONFIG
from galaxea_a1_runtime.schema import LINGBOT_EEF_ACTION_CHANNEL_IDS
from galaxea_a1_runtime.apps.lingbot.config_runtime import bash_config
from galaxea_a1_runtime.apps.lingbot.config_schema import (
    LingBotActionModeConfig,
    LingBotConfig,
    LingBotExecutionConfig,
    LingBotObservationConfig,
    LingBotPolicyServerConfig,
    LingBotServerConfig,
    LingBotServoConfig,
    LingBotSessionConfig,
    PoseMode,
    TextEncoderDevice,
)

DEFAULT_LINGBOT_CONFIG = LINGBOT_CONFIG

__all__ = ["bash_config", "load_lingbot_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LINGBOT_CONFIG


def load_lingbot_config(path: Path, *, repo_root: Path | None = None) -> LingBotConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "system",
            "session",
            "server",
            "policy_server",
            "observations",
            "execution",
            "action",
        },
        label="LingBot config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    session = _required_table(data, "session")
    server = _required_table(data, "server")
    policy_server = _required_table(data, "policy_server")
    observations = _required_table(data, "observations")
    execution = _required_table(data, "execution")
    action = _required_table(data, "action")
    require_exact_keys(session, required={"tmux"}, label="session")
    require_exact_keys(
        server,
        required={"host", "port", "connect_timeout_s", "close_timeout_s", "prompt"},
        label="server",
    )
    require_exact_keys(
        policy_server,
        required={
            "tmux",
            "checkout",
            "python",
            "base_model",
            "checkpoint",
            "model_root",
            "save_root",
            "master_port",
            "startup_timeout_s",
            "expected_weight_size_bytes",
            "deployment_ready",
            "text_encoder_device",
            "seed",
            "height",
            "width",
            "frame_chunk_size",
            "action_per_frame",
            "attention_window",
            "guidance_scale",
            "action_guidance_scale",
            "video_inference_steps",
            "action_inference_steps",
            "snr_shift",
            "action_snr_shift",
            "q01_source",
            "q99_source",
        },
        label="policy_server",
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
            "no_kv_update",
            "max_model_calls",
            "execute_frames",
            "kv_observations_per_frame",
            "condition_on_ee_state",
            "initial_ee_pose",
            "exec_rate",
            "print_actions",
            "review_deadband_m",
        },
        label="execution",
    )
    require_exact_keys(
        action,
        required={
            "pose_mode",
            "servo_gain",
            "servo_max_extra_m",
            "servo_settle_s",
            "servo_tolerance_m",
            "servo_corrections",
            "cache_actual_feedback",
        },
        label="action",
    )
    deployment_ready = boolean(policy_server, "deployment_ready")

    config = LingBotConfig(
        path=path,
        system=system,
        session=LingBotSessionConfig(tmux=_string(session, "tmux")),
        server=LingBotServerConfig(
            host=_string(server, "host"),
            port=integer(server, "port"),
            connect_timeout_s=floating(server, "connect_timeout_s"),
            close_timeout_s=floating(server, "close_timeout_s"),
            prompt=_text(server, "prompt").strip(),
        ),
        policy_server=LingBotPolicyServerConfig(
            tmux=_string(policy_server, "tmux"),
            checkout=_resolved_path(policy_server, "checkout", repo_root),
            python=_resolved_path(policy_server, "python", repo_root),
            base_model=_resolved_path(policy_server, "base_model", repo_root),
            checkpoint=_resolved_path(policy_server, "checkpoint", repo_root),
            model_root=_resolved_path(policy_server, "model_root", repo_root),
            save_root=_resolved_path(policy_server, "save_root", repo_root),
            master_port=integer(policy_server, "master_port"),
            startup_timeout_s=floating(policy_server, "startup_timeout_s"),
            expected_weight_size_bytes=integer(
                policy_server, "expected_weight_size_bytes"
            ),
            text_encoder_device=_text_encoder_device(
                _string(policy_server, "text_encoder_device")
            ),
            seed=integer(policy_server, "seed"),
            height=integer(policy_server, "height"),
            width=integer(policy_server, "width"),
            frame_chunk_size=integer(policy_server, "frame_chunk_size"),
            action_per_frame=integer(policy_server, "action_per_frame"),
            attention_window=integer(policy_server, "attention_window"),
            guidance_scale=floating(policy_server, "guidance_scale"),
            action_guidance_scale=floating(policy_server, "action_guidance_scale"),
            video_inference_steps=integer(policy_server, "video_inference_steps"),
            action_inference_steps=integer(policy_server, "action_inference_steps"),
            snr_shift=floating(policy_server, "snr_shift"),
            action_snr_shift=floating(policy_server, "action_snr_shift"),
            q01_source=float_tuple(policy_server, "q01_source"),
            q99_source=float_tuple(policy_server, "q99_source"),
            deployment_ready=deployment_ready,
        ),
        execution=LingBotExecutionConfig(
            execute=boolean(execution, "execute"),
            step_mode=boolean(execution, "step_mode"),
            step_actions=boolean(execution, "step_actions"),
            no_kv_update=boolean(execution, "no_kv_update"),
            max_model_calls=integer(execution, "max_model_calls"),
            execute_frames=integer(execution, "execute_frames"),
            kv_observations_per_frame=integer(execution, "kv_observations_per_frame"),
            condition_on_ee_state=boolean(execution, "condition_on_ee_state"),
            initial_ee_pose=_empty_float_tuple_as_none(execution, "initial_ee_pose"),
            exec_rate=floating(execution, "exec_rate"),
            print_actions=boolean(execution, "print_actions"),
            review_deadband_m=floating(execution, "review_deadband_m"),
        ),
        observations=LingBotObservationConfig(
            front_key=_string(observations, "front_key"),
            wrist_key=_string(observations, "wrist_key"),
        ),
        action=LingBotActionModeConfig(
            pose_mode=_pose_mode(_string(action, "pose_mode")),
        ),
        servo=LingBotServoConfig(
            gain=floating(action, "servo_gain"),
            max_extra_m=floating(action, "servo_max_extra_m"),
            settle_s=floating(action, "servo_settle_s"),
            tolerance_m=floating(action, "servo_tolerance_m"),
            corrections=integer(action, "servo_corrections"),
            cache_actual_feedback=boolean(action, "cache_actual_feedback"),
        ),
    )
    validate_lingbot_config(config)
    return config


def validate_lingbot_config(config: LingBotConfig) -> None:
    if not 1 <= config.server.port <= 65535:
        raise ValueError("server.port must be in [1, 65535]")
    if min(config.server.connect_timeout_s, config.server.close_timeout_s) <= 0:
        raise ValueError("server connection timeouts must be positive")
    policy = config.policy_server
    if not 1 <= policy.master_port <= 65535 or policy.startup_timeout_s <= 0:
        raise ValueError(
            "policy_server master_port must be in [1, 65535] and startup timeout must be positive"
        )
    if policy.expected_weight_size_bytes < 0:
        raise ValueError(
            "policy_server.expected_weight_size_bytes must be non-negative"
        )
    if (
        min(
            policy.height,
            policy.width,
            policy.frame_chunk_size,
            policy.action_per_frame,
        )
        <= 0
    ):
        raise ValueError("policy_server dimensions must be positive")
    if (
        policy.attention_window <= 0
        or policy.video_inference_steps <= 0
        or policy.action_inference_steps <= 0
    ):
        raise ValueError("policy_server inference settings must be positive")
    if policy.deployment_ready:
        if not config.server.prompt:
            raise ValueError(
                "deployment-ready LingBot config requires a real server.prompt"
            )
        if policy.expected_weight_size_bytes <= 0:
            raise ValueError(
                "deployment-ready LingBot config requires expected_weight_size_bytes"
            )
        if not policy.q01_source or not policy.q99_source:
            raise ValueError(
                "deployment-ready LingBot config requires real q01/q99 statistics"
            )
    elif policy.q01_source or policy.q99_source:
        raise ValueError(
            "unready LingBot config must keep q01_source/q99_source empty; "
            "do not use numeric placeholders"
        )
    if policy.q01_source and (
        len(LINGBOT_EEF_ACTION_CHANNEL_IDS) != len(policy.q01_source)
        or len(policy.q01_source) != len(policy.q99_source)
    ):
        raise ValueError(
            "policy_server action channels and quantiles must have equal lengths"
        )
    if any(
        lo >= hi for lo, hi in zip(policy.q01_source, policy.q99_source, strict=True)
    ):
        raise ValueError(
            "policy_server q01_source values must be lower than q99_source"
        )
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.execute_frames <= 0:
        raise ValueError("execution.execute_frames must be positive")
    if config.execution.kv_observations_per_frame <= 0:
        raise ValueError("execution.kv_observations_per_frame must be positive")
    if policy.action_per_frame % config.execution.kv_observations_per_frame:
        raise ValueError(
            "policy_server.action_per_frame must be divisible by "
            "execution.kv_observations_per_frame"
        )
    if (
        config.execution.initial_ee_pose is not None
        and len(config.execution.initial_ee_pose) != 8
    ):
        raise ValueError("execution.initial_ee_pose must contain 8 values")
    if config.execution.exec_rate <= 0:
        raise ValueError("execution.exec_rate must be positive")
    if config.execution.review_deadband_m < 0:
        raise ValueError("execution.review_deadband_m must be non-negative")
    if config.execution.execute and not policy.deployment_ready:
        raise ValueError(
            "execution.execute requires policy_server.deployment_ready=true"
        )
    if config.system.cameras.front.backend != "realsense":
        raise ValueError("LingBot front camera must use the RealSense backend")
    if config.servo.gain <= 0 or config.servo.tolerance_m <= 0:
        raise ValueError("servo gain and tolerance must be positive")
    if config.servo.max_extra_m < 0 or config.servo.settle_s < 0:
        raise ValueError("servo max_extra_m and settle_s must be non-negative")
    if config.servo.corrections < 0:
        raise ValueError("servo.corrections must be >= 0")


def _pose_mode(value: str) -> PoseMode:
    if value not in ("absolute", "episode-relative"):
        raise ValueError(f"unsupported eef.action_pose_mode: {value!r}")
    return value


def _text_encoder_device(value: str) -> TextEncoderDevice:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"unsupported policy_server.text_encoder_device: {value!r}")
    return value


def _resolved_path(data: dict[str, Any], key: str, repo_root: Path) -> Path:
    return _repo_path(repo_root, _string(data, key))


def _empty_float_tuple_as_none(
    data: dict[str, Any], key: str
) -> tuple[float, ...] | None:
    values = float_tuple(data, key)
    return values or None


def main(argv: list[str] | None = None) -> int:
    return run_config_renderer(
        argv,
        description="Read the tracked A1 LingBot deployment config.",
        default_config=DEFAULT_LINGBOT_CONFIG,
        load_config=load_lingbot_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
