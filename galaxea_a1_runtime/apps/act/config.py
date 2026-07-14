"""Git-tracked ACT joint-state runtime configuration."""

from __future__ import annotations

import argparse
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.hardware.image_geometry import (
    ImageRoi,
    parse_optional_image_roi,
)
from galaxea_a1_runtime.hardware.web_preview import (
    WebPreviewConfig,
    parse_web_preview_config,
    web_preview_argv,
)

DEFAULT_ACT_CONFIG = Path("configs/inference/a1_act_joint.toml")


@dataclass(frozen=True)
class ActSessionConfig:
    tmux: str


@dataclass(frozen=True)
class ActHostConfig:
    image: str
    a1_serial: str
    prefix: str
    run_dir: str


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
class ActTopicsConfig:
    joint_states: str
    target: str
    staged_command: str
    motion_enable: str
    relay_status: str
    gripper_command: str
    gripper_feedback: str


@dataclass(frozen=True)
class ActRelayConfig:
    enable_timeout_s: float
    max_status_age_s: float


@dataclass(frozen=True)
class ActSafetyConfig:
    target_joint_names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    max_joint_action_step_rad: float
    max_first_target_delta_rad: float
    initial_alignment_tolerance_rad: float
    state_timeout_s: float
    max_feedback_age_s: float
    max_camera_age_s: float


@dataclass(frozen=True)
class ActGripperConfig:
    command_mode: str
    stroke_min_mm: float
    stroke_max_mm: float
    command_open_threshold: float
    feedback_open_threshold_mm: float


@dataclass(frozen=True)
class ActCameraConfig:
    width: int
    height: int
    fps: int
    warmup_frames: int
    front_serial: str
    front_auto_exposure: bool
    front_exposure: int
    front_gain: int
    front_auto_white_balance: bool
    front_white_balance: int
    front_crop: ImageRoi | None
    wrist_backend: str
    wrist_serial: str
    wrist_device: str
    wrist_backend_api: str
    wrist_pixel_format: str


@dataclass(frozen=True)
class ActConfig:
    path: Path
    session: ActSessionConfig
    host: ActHostConfig
    policy: ActPolicyConfig
    execution: ActExecutionConfig
    topics: ActTopicsConfig
    relay: ActRelayConfig
    safety: ActSafetyConfig
    gripper: ActGripperConfig
    cameras: ActCameraConfig
    web_preview: WebPreviewConfig


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ACT_CONFIG


def load_act_config(path: Path, *, repo_root: Path | None = None) -> ActConfig:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    path = path.resolve()
    repo_root = repo_root.resolve() if repo_root is not None else path.parents[2]
    data = tomllib.loads(path.read_text())

    session = _required_table(data, "session")
    host = _required_table(data, "host")
    policy = _required_table(data, "policy")
    execution = _required_table(data, "execution")
    topics = _required_table(data, "topics")
    relay = _required_table(data, "relay")
    safety = _required_table(data, "safety")
    gripper = _required_table(data, "gripper")
    cameras = _required_table(data, "cameras")
    front = _required_table(cameras, "front")
    wrist = _required_table(cameras, "wrist")
    camera_width = int(cameras.get("width", 640))
    camera_height = int(cameras.get("height", 480))

    config = ActConfig(
        path=path,
        session=ActSessionConfig(tmux=_string(session, "tmux")),
        host=ActHostConfig(
            image=_string(host, "image"),
            a1_serial=_string(host, "a1_serial"),
            prefix=_string(host, "prefix"),
            run_dir=_string(host, "run_dir"),
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
        topics=ActTopicsConfig(
            joint_states=_string(topics, "joint_states"),
            target=_string(topics, "target"),
            staged_command=_string(topics, "staged_command"),
            motion_enable=_string(topics, "motion_enable"),
            relay_status=_string(topics, "relay_status"),
            gripper_command=_string(topics, "gripper_command"),
            gripper_feedback=_string(topics, "gripper_feedback"),
        ),
        relay=ActRelayConfig(
            enable_timeout_s=float(relay.get("enable_timeout_s", 2.0)),
            max_status_age_s=float(relay.get("max_status_age_s", 1.0)),
        ),
        safety=ActSafetyConfig(
            target_joint_names=_string_tuple(safety, "target_joint_names", 6),
            lower_limits=_float_tuple(safety, "lower_limits", 6),
            upper_limits=_float_tuple(safety, "upper_limits", 6),
            max_joint_action_step_rad=float(safety.get("max_joint_action_step_rad", 0.25)),
            max_first_target_delta_rad=float(safety.get("max_first_target_delta_rad", 0.25)),
            initial_alignment_tolerance_rad=float(safety.get("initial_alignment_tolerance_rad", 0.05)),
            state_timeout_s=float(safety.get("state_timeout_s", 10.0)),
            max_feedback_age_s=float(safety.get("max_feedback_age_s", 0.5)),
            max_camera_age_s=float(safety.get("max_camera_age_s", 0.5)),
        ),
        gripper=ActGripperConfig(
            command_mode=str(gripper.get("command_mode", "binary")),
            stroke_min_mm=float(gripper.get("stroke_min_mm", 0.0)),
            stroke_max_mm=float(gripper.get("stroke_max_mm", 200.0)),
            command_open_threshold=float(gripper.get("command_open_threshold", 0.5)),
            feedback_open_threshold_mm=float(gripper.get("feedback_open_threshold_mm", 30.0)),
        ),
        cameras=ActCameraConfig(
            width=camera_width,
            height=camera_height,
            fps=int(cameras.get("fps", 30)),
            warmup_frames=int(cameras.get("warmup_frames", 20)),
            front_serial=str(front.get("serial", "")),
            front_auto_exposure=bool(front.get("auto_exposure", True)),
            front_exposure=int(front.get("exposure", 140)),
            front_gain=int(front.get("gain", 32)),
            front_auto_white_balance=bool(front.get("auto_white_balance", True)),
            front_white_balance=int(front.get("white_balance", 4600)),
            front_crop=parse_optional_image_roi(
                front,
                image_width=camera_width,
                image_height=camera_height,
                label="cameras.front crop",
                require_square=True,
            ),
            wrist_backend=str(wrist.get("backend", "v4l2")),
            wrist_serial=str(wrist.get("serial", "")),
            wrist_device=str(wrist.get("device", "")),
            wrist_backend_api=str(wrist.get("backend_api", "v4l2")),
            wrist_pixel_format=str(wrist.get("pixel_format", "")),
        ),
        web_preview=parse_web_preview_config(
            data.get("web_preview", {}) if isinstance(data.get("web_preview", {}), dict) else {},
            repo_root=repo_root,
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
    if config.relay.enable_timeout_s <= 0 or config.relay.max_status_age_s <= 0:
        raise ValueError("relay timeouts must be positive")
    if config.cameras.width <= 0 or config.cameras.height <= 0 or config.cameras.fps <= 0:
        raise ValueError("camera width/height/fps must be positive")
    if config.cameras.warmup_frames < 0:
        raise ValueError("cameras.warmup_frames must be non-negative")
    if not config.cameras.front_serial:
        raise ValueError("cameras.front.serial is required")
    if config.cameras.front_crop is None:
        raise ValueError("cameras.front crop must be enabled for the inference input contract")
    if config.cameras.wrist_backend not in {"realsense", "v4l2"}:
        raise ValueError("cameras.wrist.backend must be 'realsense' or 'v4l2'")
    if config.cameras.wrist_backend == "realsense" and not config.cameras.wrist_serial:
        raise ValueError("cameras.wrist.serial is required for the RealSense backend")
    if config.cameras.wrist_backend == "v4l2" and not config.cameras.wrist_device:
        raise ValueError("cameras.wrist.device is required for the V4L2 backend")
    for name, value in config.topics.__dict__.items():
        if not value.startswith("/"):
            raise ValueError(f"topics.{name} must be an absolute ROS topic: {value!r}")
    if len(set(config.safety.target_joint_names)) != len(config.safety.target_joint_names):
        raise ValueError("safety.target_joint_names must not contain duplicates")
    if any(lo >= hi for lo, hi in zip(config.safety.lower_limits, config.safety.upper_limits, strict=True)):
        raise ValueError("safety.lower_limits must be below upper_limits")
    for label, value in (
        ("safety.max_joint_action_step_rad", config.safety.max_joint_action_step_rad),
        ("safety.max_first_target_delta_rad", config.safety.max_first_target_delta_rad),
        ("safety.initial_alignment_tolerance_rad", config.safety.initial_alignment_tolerance_rad),
        ("safety.state_timeout_s", config.safety.state_timeout_s),
        ("safety.max_feedback_age_s", config.safety.max_feedback_age_s),
        ("safety.max_camera_age_s", config.safety.max_camera_age_s),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if config.gripper.stroke_max_mm <= config.gripper.stroke_min_mm:
        raise ValueError("gripper.stroke_max_mm must be greater than stroke_min_mm")
    if config.gripper.command_mode not in {"binary", "continuous"}:
        raise ValueError("gripper.command_mode must be 'binary' or 'continuous'")
    if not 0.0 < config.gripper.command_open_threshold < 1.0:
        raise ValueError("gripper.command_open_threshold must be between 0 and 1")
    if not config.gripper.stroke_min_mm < config.gripper.feedback_open_threshold_mm < config.gripper.stroke_max_mm:
        raise ValueError("gripper.feedback_open_threshold_mm must be inside the physical stroke range")


def bridge_argv(config: ActConfig) -> list[str]:
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
        config.topics.joint_states,
        "--target-topic",
        config.topics.target,
        "--staged-command-topic",
        config.topics.staged_command,
        "--motion-enable-topic",
        config.topics.motion_enable,
        "--relay-status-topic",
        config.topics.relay_status,
        "--gripper-command-topic",
        config.topics.gripper_command,
        "--gripper-feedback-topic",
        config.topics.gripper_feedback,
        "--gripper-command-mode",
        config.gripper.command_mode,
        "--relay-enable-timeout",
        _num(config.relay.enable_timeout_s),
        "--max-relay-status-age",
        _num(config.relay.max_status_age_s),
        "--target-joint-names",
        *config.safety.target_joint_names,
        "--lower-limits",
        *(_num(value) for value in config.safety.lower_limits),
        "--upper-limits",
        *(_num(value) for value in config.safety.upper_limits),
        "--max-joint-action-step-rad",
        _num(config.safety.max_joint_action_step_rad),
        "--max-first-target-delta-rad",
        _num(config.safety.max_first_target_delta_rad),
        "--initial-alignment-tolerance",
        _num(config.safety.initial_alignment_tolerance_rad),
        "--state-timeout",
        _num(config.safety.state_timeout_s),
        "--max-feedback-age",
        _num(config.safety.max_feedback_age_s),
        "--max-camera-age",
        _num(config.safety.max_camera_age_s),
        "--gripper-stroke-min",
        _num(config.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        _num(config.gripper.stroke_max_mm),
        "--gripper-command-open-threshold",
        _num(config.gripper.command_open_threshold),
        "--gripper-feedback-open-threshold-mm",
        _num(config.gripper.feedback_open_threshold_mm),
        "--cam-width",
        str(config.cameras.width),
        "--cam-height",
        str(config.cameras.height),
        "--cam-fps",
        str(config.cameras.fps),
        "--camera-warmup-frames",
        str(config.cameras.warmup_frames),
        _bool_flag("cam0-auto-exposure", config.cameras.front_auto_exposure),
        "--cam0-exposure",
        str(config.cameras.front_exposure),
        "--cam0-gain",
        str(config.cameras.front_gain),
        _bool_flag("cam0-auto-white-balance", config.cameras.front_auto_white_balance),
        "--cam0-white-balance",
        str(config.cameras.front_white_balance),
        _bool_flag("cam0-crop-enabled", config.cameras.front_crop is not None),
        "--cam1-device",
        config.cameras.wrist_device,
        "--cam1-backend",
        config.cameras.wrist_backend,
        "--cam1-serial",
        config.cameras.wrist_serial,
        "--cam1-backend-api",
        config.cameras.wrist_backend_api,
        "--cam1-pixel-format",
        config.cameras.wrist_pixel_format,
        *web_preview_argv(config.web_preview),
    ]
    if config.cameras.front_serial:
        args.extend(["--cam0-serial", config.cameras.front_serial])
    if config.cameras.front_crop is not None:
        args.extend(
            [
                "--cam0-crop-x",
                str(config.cameras.front_crop.x),
                "--cam0-crop-y",
                str(config.cameras.front_crop.y),
                "--cam0-crop-width",
                str(config.cameras.front_crop.width),
                "--cam0-crop-height",
                str(config.cameras.front_crop.height),
            ]
        )
    return args


def bash_config(config: ActConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SESSION", config.session.tmux),
        _assign("IMAGE", config.host.image),
        _assign("SERIAL", config.host.a1_serial),
        _assign("PREFIX", config.host.prefix),
        _assign("RUN_DIR", config.host.run_dir),
        _assign("CHECKPOINT", str(config.policy.checkpoint)),
        _assign("DEPLOYMENT_READY", "1" if config.policy.deployment_ready else "0"),
        _assign("WRIST_BACKEND", config.cameras.wrist_backend),
        _assign("WRIST_SERIAL", config.cameras.wrist_serial),
        _assign("WRIST_CAMERA", config.cameras.wrist_device),
        _assign("TARGET_TOPIC", config.topics.target),
        _assign("STAGED_TOPIC", config.topics.staged_command),
        _assign("RELAY_ENABLE_TOPIC", config.topics.motion_enable),
        _assign("RELAY_STATUS_TOPIC", config.topics.relay_status),
        _array("BRIDGE_ARGS", bridge_argv(config)),
    ]
    return "\n".join(lines)


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{key}] table")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string value: {key}")
    return value


def _string_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a string list")
    out = tuple(str(item) for item in value)
    if len(out) != expected_len or any(not item for item in out):
        raise ValueError(f"{key} expects {expected_len} non-empty strings")
    return out


def _float_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    out = tuple(float(item) for item in value)
    if len(out) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(out)}")
    return out


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _num(value: float) -> str:
    return f"{float(value):g}"


def _assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _array(name: str, values: list[str]) -> str:
    return f"{name}=({' '.join(shlex.quote(value) for value in values)})"


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
