"""Git-tracked ACT joint-state runtime configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    bool_flag as _bool_flag,
    load_toml,
    number as _num,
    referenced_config,
    repo_path as _repo_path,
    required_table as _required_table,
    shell_array as _array,
    shell_assign as _assign,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import SystemConfig, load_system_config
from galaxea_a1_runtime.hardware.web_preview import web_preview_argv

DEFAULT_ACT_CONFIG = Path("configs/deployments/act_joint.toml")


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


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ACT_CONFIG


def load_act_config(path: Path, *, repo_root: Path | None = None) -> ActConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)

    session = _required_table(data, "session")
    runtime = _required_table(data, "runtime")
    policy = _required_table(data, "policy")
    execution = _required_table(data, "execution")
    config = ActConfig(
        path=path,
        system=system,
        session=ActSessionConfig(tmux=_string(session, "tmux")),
        runtime=ActRuntimeConfig(
            prefix=_string(runtime, "prefix"),
        ),
        policy=ActPolicyConfig(
            checkpoint=_repo_path(repo_root, _string(policy, "checkpoint")),
            device=_string(policy, "device"),
            disable_backbone_download=bool(policy.get("disable_backbone_download", True)),
            deployment_ready=bool(policy.get("deployment_ready", False)),
        ),
        execution=ActExecutionConfig(
            execute=bool(execution.get("execute", False)),
            step_mode=bool(execution.get("step_mode", True)),
            execute_steps_per_inference=int(execution.get("execute_steps_per_inference", 8)),
            control_hz=float(execution.get("control_hz", 30.0)),
            max_model_calls=int(execution.get("max_model_calls", 0)),
            print_actions=bool(execution.get("print_actions", True)),
            preview_steps=int(execution.get("preview_steps", 5)),
        ),
    )
    validate_act_config(config)
    return config


def validate_act_config(config: ActConfig) -> None:
    if config.execution.execute_steps_per_inference <= 0:
        raise ValueError("execution.execute_steps_per_inference must be positive")
    if config.execution.control_hz <= 0:
        raise ValueError("execution.control_hz must be positive")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.preview_steps <= 0:
        raise ValueError("execution.preview_steps must be positive")


def bridge_argv(config: ActConfig) -> list[str]:
    system = config.system
    topics = system.topics
    safety = system.joint_safety
    front = system.cameras.front
    wrist = system.cameras.wrist
    args = [
        "--checkpoint",
        str(config.policy.checkpoint),
        "--device",
        config.policy.device,
        _bool_flag("disable-backbone-download", config.policy.disable_backbone_download),
        _bool_flag("execute", config.execution.execute),
        _bool_flag("step-mode", config.execution.step_mode),
        "--execute-steps-per-inference",
        str(config.execution.execute_steps_per_inference),
        "--control-hz",
        _num(config.execution.control_hz),
        "--max-model-calls",
        str(config.execution.max_model_calls),
        _bool_flag("print-actions", config.execution.print_actions),
        "--preview-steps",
        str(config.execution.preview_steps),
        "--joint-states-topic",
        topics.joint_states,
        "--target-topic",
        topics.joint_target,
        "--staged-command-topic",
        topics.staged_command,
        "--motion-enable-topic",
        topics.motion_enable,
        "--relay-status-topic",
        topics.relay_status,
        "--gripper-target-topic",
        topics.gripper_target,
        "--gripper-feedback-topic",
        topics.gripper_feedback,
        "--relay-enable-timeout",
        _num(system.relay.enable_timeout_s),
        "--max-relay-status-age",
        _num(system.relay.max_status_age_s),
        "--target-joint-names",
        *safety.names,
        "--lower-limits",
        *(_num(value) for value in safety.lower_limits),
        "--upper-limits",
        *(_num(value) for value in safety.upper_limits),
        _bool_flag("action-step-guard-enabled", safety.action_step_guard_enabled),
        "--max-joint-action-step-rad",
        _num(safety.max_action_step_rad),
        "--max-first-target-delta-rad",
        _num(safety.max_first_target_delta_rad),
        "--initial-alignment-tolerance",
        _num(safety.initial_alignment_tolerance_rad),
        "--state-timeout",
        _num(safety.state_timeout_s),
        "--max-feedback-age",
        _num(safety.max_feedback_age_s),
        "--max-camera-age",
        _num(system.cameras.max_age_s),
        "--gripper-stroke-min",
        _num(system.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        _num(system.gripper.stroke_max_mm),
        "--cam-width",
        str(front.width),
        "--cam-height",
        str(front.height),
        "--cam-fps",
        str(front.fps),
        "--camera-warmup-frames",
        str(system.cameras.warmup_frames),
        _bool_flag("cam0-auto-exposure", front.auto_exposure),
        "--cam0-exposure",
        str(front.exposure),
        "--cam0-gain",
        str(front.gain),
        _bool_flag("cam0-auto-white-balance", front.auto_white_balance),
        "--cam0-white-balance",
        str(front.white_balance),
        _bool_flag("cam0-crop-enabled", front.crop is not None),
        "--cam1-device",
        wrist.device,
        "--cam1-backend",
        wrist.backend,
        "--cam1-serial",
        wrist.serial,
        "--cam1-backend-api",
        wrist.backend_api,
        "--cam1-pixel-format",
        wrist.pixel_format,
        *web_preview_argv(system.web_preview),
    ]
    if front.serial:
        args.extend(["--cam0-serial", front.serial])
    if front.crop is not None:
        args.extend(
            [
                "--cam0-crop-x",
                str(front.crop.x),
                "--cam0-crop-y",
                str(front.crop.y),
                "--cam0-crop-width",
                str(front.crop.width),
                "--cam0-crop-height",
                str(front.crop.height),
            ]
        )
    return args


def bash_config(config: ActConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SYSTEM_CONFIG_PATH", str(config.system.path)),
        _assign("SESSION", config.session.tmux),
        _assign("PREFIX", config.runtime.prefix),
        _assign("CHECKPOINT", str(config.policy.checkpoint)),
        _assign("DEPLOYMENT_READY", "1" if config.policy.deployment_ready else "0"),
        _assign("WRIST_BACKEND", config.system.cameras.wrist.backend),
        _assign("WRIST_SERIAL", config.system.cameras.wrist.serial),
        _assign("WRIST_CAMERA", config.system.cameras.wrist.device),
        _array("BRIDGE_ARGS", bridge_argv(config)),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read A1 ACT joint inference TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shell", action="store_true", help="Emit bash assignments for a1_act_joint_runtime.sh")
    args = parser.parse_args(argv)

    config = load_act_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
