"""Git-tracked LingBot-VA runtime configuration."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    load_toml,
    referenced_config,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.apps.lingbot.config_runtime import bash_config, bridge_argv
from galaxea_a1_runtime.apps.lingbot.config_schema import (
    LingBotActionConfig,
    LingBotConfig,
    LingBotExecutionConfig,
    LingBotObservationConfig,
    LingBotPolicyServerConfig,
    LingBotServerConfig,
    LingBotServoConfig,
    LingBotSessionConfig,
    OrientationMode,
    PoseMode,
    TextEncoderDevice,
)

DEFAULT_LINGBOT_CONFIG = Path("configs/deployments/lingbot_va.toml")

__all__ = ["bash_config", "bridge_argv", "load_lingbot_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LINGBOT_CONFIG


def load_lingbot_config(path: Path, *, repo_root: Path | None = None) -> LingBotConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    session = _required_table(data, "session")
    server = _required_table(data, "server")
    policy_server = _required_table(data, "policy_server")
    observations = _required_table(data, "observations")
    execution = _required_table(data, "execution")
    action = _required_table(data, "action")
    deployment_ready = bool(policy_server.get("deployment_ready", False))

    config = LingBotConfig(
        path=path,
        system=system,
        session=LingBotSessionConfig(tmux=_string(session, "tmux")),
        server=LingBotServerConfig(
            host=_string(server, "host"),
            port=int(server.get("port", 1106)),
            prompt=_string(server, "prompt"),
        ),
        policy_server=LingBotPolicyServerConfig(
            tmux=_string(policy_server, "tmux"),
            checkout=_resolved_path(policy_server, "checkout", repo_root),
            python=_resolved_path(policy_server, "python", repo_root),
            base_model=_resolved_path(policy_server, "base_model", repo_root),
            checkpoint=_resolved_path(policy_server, "checkpoint", repo_root),
            model_root=_resolved_path(policy_server, "model_root", repo_root),
            save_root=_resolved_path(policy_server, "save_root", repo_root),
            master_port=int(policy_server.get("master_port", 29501)),
            startup_timeout_s=float(policy_server.get("startup_timeout_s", 120.0)),
            expected_weight_size_bytes=int(
                policy_server.get("expected_weight_size_bytes", 0)
            ),
            text_encoder_device=_text_encoder_device(
                str(policy_server.get("text_encoder_device", "cpu"))
            ),
            seed=int(policy_server.get("seed", 42)),
            height=int(policy_server.get("height", 256)),
            width=int(policy_server.get("width", 256)),
            frame_chunk_size=int(policy_server.get("frame_chunk_size", 4)),
            action_per_frame=int(policy_server.get("action_per_frame", 4)),
            attention_window=int(policy_server.get("attention_window", 30)),
            guidance_scale=float(policy_server.get("guidance_scale", 5.0)),
            action_guidance_scale=float(
                policy_server.get("action_guidance_scale", 1.0)
            ),
            video_inference_steps=int(policy_server.get("video_inference_steps", 5)),
            action_inference_steps=int(policy_server.get("action_inference_steps", 10)),
            snr_shift=float(policy_server.get("snr_shift", 5.0)),
            action_snr_shift=float(policy_server.get("action_snr_shift", 1.0)),
            used_action_channel_ids=_int_tuple(
                policy_server, "used_action_channel_ids"
            ),
            q01_source=_float_tuple(policy_server, "q01_source"),
            q99_source=_float_tuple(policy_server, "q99_source"),
            deployment_ready=deployment_ready,
        ),
        execution=LingBotExecutionConfig(
            execute=bool(execution.get("execute", True)),
            step_mode=bool(execution.get("step_mode", True)),
            step_actions=bool(execution.get("step_actions", True)),
            no_kv_update=bool(execution.get("no_kv_update", False)),
            max_model_calls=int(execution.get("max_model_calls", 0)),
            execute_frames=int(execution.get("execute_frames", 1)),
            condition_on_ee_state=bool(execution.get("condition_on_ee_state", True)),
            initial_ee_pose=_optional_float_tuple(execution, "initial_ee_pose"),
            lingbot_frame_chunk_size=int(execution.get("lingbot_frame_chunk_size", 4)),
            lingbot_action_per_frame=int(execution.get("lingbot_action_per_frame", 20)),
            exec_rate=float(execution.get("exec_rate", 30.0)),
            print_actions=bool(execution.get("print_actions", True)),
            review_deadband_m=float(execution.get("review_deadband_m", 0.001)),
        ),
        observations=LingBotObservationConfig(
            front_key=_string(observations, "front_key"),
            wrist_key=_string(observations, "wrist_key"),
        ),
        action=LingBotActionConfig(
            pose_mode=_pose_mode(_string(action, "pose_mode")),
        ),
        servo=LingBotServoConfig(
            gain=float(action.get("servo_gain", 1.0)),
            max_extra_m=float(action.get("servo_max_extra_m", 0.04)),
            settle_s=float(action.get("servo_settle_s", 0.0)),
            tolerance_m=float(action.get("servo_tolerance_m", 0.01)),
            corrections=int(action.get("servo_corrections", 0)),
            cache_actual_feedback=bool(action.get("cache_actual_feedback", False)),
        ),
    )
    validate_lingbot_config(config)
    return config


def validate_lingbot_config(config: LingBotConfig) -> None:
    if config.server.port <= 0:
        raise ValueError("server.port must be positive")
    policy = config.policy_server
    if policy.master_port <= 0 or policy.startup_timeout_s <= 0:
        raise ValueError(
            "policy_server master port and startup timeout must be positive"
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
        if config.server.prompt.startswith("REPLACE_WITH_"):
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
        len(policy.used_action_channel_ids) != len(policy.q01_source)
        or len(policy.q01_source) != len(policy.q99_source)
    ):
        raise ValueError(
            "policy_server action channels and quantiles must have equal lengths"
        )
    if len(set(policy.used_action_channel_ids)) != len(policy.used_action_channel_ids):
        raise ValueError("policy_server.used_action_channel_ids must be unique")
    if any(channel < 0 or channel >= 30 for channel in policy.used_action_channel_ids):
        raise ValueError("policy_server action channels must be in [0, 30)")
    if any(
        not math.isfinite(value) for value in (*policy.q01_source, *policy.q99_source)
    ):
        raise ValueError("policy_server quantiles must be finite")
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
    if (
        config.execution.initial_ee_pose is not None
        and len(config.execution.initial_ee_pose) != 8
    ):
        raise ValueError("execution.initial_ee_pose must contain 8 values")
    if (
        config.execution.lingbot_frame_chunk_size <= 0
        or config.execution.lingbot_action_per_frame <= 0
    ):
        raise ValueError("LingBot frame/action dimensions must be positive")
    if config.execution.lingbot_frame_chunk_size != policy.frame_chunk_size:
        raise ValueError("execution and policy_server frame_chunk_size must match")
    if config.execution.lingbot_action_per_frame != policy.action_per_frame:
        raise ValueError("execution and policy_server action_per_frame must match")
    if config.execution.exec_rate <= 0:
        raise ValueError("execution.exec_rate must be positive")
    if config.system.cameras.front.backend != "realsense":
        raise ValueError("LingBot front camera must use the RealSense backend")
    if config.servo.gain <= 0:
        raise ValueError("servo.gain must be positive")
    if config.servo.corrections < 0:
        raise ValueError("servo.corrections must be >= 0")


def _orientation_mode(value: str) -> OrientationMode:
    if value not in ("hold-current", "model-quat"):
        raise ValueError(f"unsupported eef.orientation_mode: {value!r}")
    return value


def _pose_mode(value: str) -> PoseMode:
    if value not in ("absolute", "episode-relative"):
        raise ValueError(f"unsupported eef.action_pose_mode: {value!r}")
    return value


def _text_encoder_device(value: str) -> TextEncoderDevice:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"unsupported policy_server.text_encoder_device: {value!r}")
    return value


def _float_tuple(data: dict[str, Any], key: str) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    return tuple(float(item) for item in value)


def _int_tuple(data: dict[str, Any], key: str) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty integer list")
    return tuple(int(item) for item in value)


def _resolved_path(data: dict[str, Any], key: str, repo_root: Path) -> Path:
    return _repo_path(repo_root, _string(data, key))


def _optional_float_tuple(data: dict[str, Any], key: str) -> tuple[float, ...] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    return tuple(float(item) for item in value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read A1 LingBot TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Emit bash assignments for a1_lingbot_runtime.sh",
    )
    args = parser.parse_args(argv)

    config = load_lingbot_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
