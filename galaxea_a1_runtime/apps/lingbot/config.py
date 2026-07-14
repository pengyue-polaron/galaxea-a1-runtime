"""Git-tracked LingBot-VA runtime configuration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
from galaxea_a1_runtime.hardware.web_preview import (
    WebPreviewConfig,
    web_preview_argv,
)

OrientationMode = Literal["hold-current", "model-quat"]
PoseMode = Literal["absolute", "episode-relative"]
GripperCommandMode = Literal["binary", "continuous"]
TextEncoderDevice = Literal["cpu", "cuda"]

DEFAULT_LINGBOT_CONFIG = Path("configs/deployments/lingbot_va.toml")


@dataclass(frozen=True)
class LingBotSessionConfig:
    tmux: str


@dataclass(frozen=True)
class LingBotServerConfig:
    host: str
    port: int
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
    used_action_channel_ids: tuple[int, ...]
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
    condition_on_ee_state: bool
    initial_ee_pose: tuple[float, ...] | None
    lingbot_frame_chunk_size: int
    lingbot_action_per_frame: int
    exec_rate: float
    print_actions: bool
    review_deadband_m: float


@dataclass(frozen=True)
class LingBotTopicsConfig:
    state_pose: str
    state_gripper: str
    cmd_pose: str
    cmd_gripper: str
    motion_enable: str
    relay_status: str
    staged_command: str


@dataclass(frozen=True)
class LingBotRelayConfig:
    enable_timeout_s: float
    max_status_age_s: float


@dataclass(frozen=True)
class LingBotEefConfig:
    command_frame: str
    action_pose_mode: PoseMode
    orientation_mode: OrientationMode
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    min_quat_norm: float
    max_feedback_age_s: float
    feedback_wait_timeout_s: float


@dataclass(frozen=True)
class LingBotServoConfig:
    gain: float
    max_extra_m: float
    settle_s: float
    tolerance_m: float
    corrections: int
    cache_actual_feedback: bool


@dataclass(frozen=True)
class LingBotGripperConfig:
    stroke_scale_mm: float
    stroke_offset_mm: float
    stroke_min_mm: float
    stroke_max_mm: float
    command_mode: GripperCommandMode
    command_open_threshold: float
    feedback_open_threshold_mm: float


@dataclass(frozen=True)
class LingBotCameraConfig:
    width: int
    height: int
    fps: int
    max_camera_age_s: float
    front_serial: str
    front_auto_exposure: bool
    front_exposure: int
    front_gain: int
    front_auto_white_balance: bool
    front_white_balance: int
    front_observation_key: str
    front_crop: ImageRoi | None
    wrist_backend: str
    wrist_serial: str
    wrist_device: str
    wrist_backend_api: str
    wrist_observation_key: str


@dataclass(frozen=True)
class LingBotConfig:
    path: Path
    system: SystemConfig
    session: LingBotSessionConfig
    server: LingBotServerConfig
    policy_server: LingBotPolicyServerConfig
    execution: LingBotExecutionConfig
    topics: LingBotTopicsConfig
    relay: LingBotRelayConfig
    eef: LingBotEefConfig
    servo: LingBotServoConfig
    gripper: LingBotGripperConfig
    cameras: LingBotCameraConfig
    web_preview: WebPreviewConfig


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
    gripper = _required_table(data, "gripper_policy")
    front = system.cameras.front
    wrist = system.cameras.wrist

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
            expected_weight_size_bytes=int(policy_server.get("expected_weight_size_bytes", 0)),
            text_encoder_device=_text_encoder_device(str(policy_server.get("text_encoder_device", "cpu"))),
            seed=int(policy_server.get("seed", 42)),
            height=int(policy_server.get("height", 256)),
            width=int(policy_server.get("width", 256)),
            frame_chunk_size=int(policy_server.get("frame_chunk_size", 4)),
            action_per_frame=int(policy_server.get("action_per_frame", 4)),
            attention_window=int(policy_server.get("attention_window", 30)),
            guidance_scale=float(policy_server.get("guidance_scale", 5.0)),
            action_guidance_scale=float(policy_server.get("action_guidance_scale", 1.0)),
            video_inference_steps=int(policy_server.get("video_inference_steps", 5)),
            action_inference_steps=int(policy_server.get("action_inference_steps", 10)),
            snr_shift=float(policy_server.get("snr_shift", 5.0)),
            action_snr_shift=float(policy_server.get("action_snr_shift", 1.0)),
            used_action_channel_ids=_int_tuple(policy_server, "used_action_channel_ids"),
            q01_source=_float_tuple_any(policy_server, "q01_source"),
            q99_source=_float_tuple_any(policy_server, "q99_source"),
            deployment_ready=bool(policy_server.get("deployment_ready", False)),
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
        topics=LingBotTopicsConfig(
            state_pose=system.topics.eef_pose,
            state_gripper=system.topics.gripper_feedback,
            cmd_pose=system.topics.eef_target,
            cmd_gripper=system.topics.gripper_command,
            motion_enable=system.topics.motion_enable,
            relay_status=system.topics.relay_status,
            staged_command=system.topics.staged_command,
        ),
        relay=LingBotRelayConfig(
            enable_timeout_s=system.relay.enable_timeout_s,
            max_status_age_s=system.relay.max_status_age_s,
        ),
        eef=LingBotEefConfig(
            command_frame=system.eef.command_frame,
            action_pose_mode=_pose_mode(_string(action, "pose_mode")),
            orientation_mode=_orientation_mode(system.eef.orientation_mode),
            xyz_min=system.eef.xyz_min,
            xyz_max=system.eef.xyz_max,
            min_quat_norm=system.eef.min_quat_norm,
            max_feedback_age_s=system.eef.max_feedback_age_s,
            feedback_wait_timeout_s=system.eef.feedback_wait_timeout_s,
        ),
        servo=LingBotServoConfig(
            gain=float(action.get("servo_gain", 1.0)),
            max_extra_m=float(action.get("servo_max_extra_m", 0.04)),
            settle_s=float(action.get("servo_settle_s", 0.0)),
            tolerance_m=float(action.get("servo_tolerance_m", 0.01)),
            corrections=int(action.get("servo_corrections", 0)),
            cache_actual_feedback=bool(action.get("cache_actual_feedback", False)),
        ),
        gripper=LingBotGripperConfig(
            stroke_scale_mm=float(gripper.get("stroke_scale_mm", 200.0)),
            stroke_offset_mm=float(gripper.get("stroke_offset_mm", 0.0)),
            stroke_min_mm=float(gripper.get("output_stroke_min_mm", 0.0)),
            stroke_max_mm=float(gripper.get("output_stroke_max_mm", 200.0)),
            command_mode=_gripper_command_mode(str(gripper.get("command_mode", "binary"))),
            command_open_threshold=float(gripper.get("command_open_threshold", 0.5)),
            feedback_open_threshold_mm=system.gripper.feedback_open_threshold_mm,
        ),
        cameras=LingBotCameraConfig(
            width=front.width,
            height=front.height,
            fps=front.fps,
            max_camera_age_s=system.cameras.max_age_s,
            front_serial=front.serial,
            front_auto_exposure=front.auto_exposure,
            front_exposure=front.exposure,
            front_gain=front.gain,
            front_auto_white_balance=front.auto_white_balance,
            front_white_balance=front.white_balance,
            front_observation_key=_string(observations, "front_key"),
            front_crop=front.crop,
            wrist_backend=wrist.backend,
            wrist_serial=wrist.serial,
            wrist_device=wrist.device,
            wrist_backend_api=wrist.backend_api,
            wrist_observation_key=_string(observations, "wrist_key"),
        ),
        web_preview=system.web_preview,
    )
    validate_lingbot_config(config)
    return config


def validate_lingbot_config(config: LingBotConfig) -> None:
    if config.server.port <= 0:
        raise ValueError("server.port must be positive")
    policy = config.policy_server
    if policy.master_port <= 0 or policy.startup_timeout_s <= 0:
        raise ValueError("policy_server master port and startup timeout must be positive")
    if policy.expected_weight_size_bytes <= 0:
        raise ValueError("policy_server.expected_weight_size_bytes must be positive")
    if min(policy.height, policy.width, policy.frame_chunk_size, policy.action_per_frame) <= 0:
        raise ValueError("policy_server dimensions must be positive")
    if policy.attention_window <= 0 or policy.video_inference_steps <= 0 or policy.action_inference_steps <= 0:
        raise ValueError("policy_server inference settings must be positive")
    if len(policy.used_action_channel_ids) != len(policy.q01_source) or len(policy.q01_source) != len(
        policy.q99_source
    ):
        raise ValueError("policy_server action channels and quantiles must have equal lengths")
    if len(set(policy.used_action_channel_ids)) != len(policy.used_action_channel_ids):
        raise ValueError("policy_server.used_action_channel_ids must be unique")
    if any(channel < 0 or channel >= 30 for channel in policy.used_action_channel_ids):
        raise ValueError("policy_server action channels must be in [0, 30)")
    if any(lo >= hi for lo, hi in zip(policy.q01_source, policy.q99_source, strict=True)):
        raise ValueError("policy_server q01_source values must be lower than q99_source")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.execute_frames <= 0:
        raise ValueError("execution.execute_frames must be positive")
    if config.execution.initial_ee_pose is not None and len(config.execution.initial_ee_pose) != 8:
        raise ValueError("execution.initial_ee_pose must contain 8 values")
    if config.execution.lingbot_frame_chunk_size <= 0 or config.execution.lingbot_action_per_frame <= 0:
        raise ValueError("LingBot frame/action dimensions must be positive")
    if config.execution.lingbot_frame_chunk_size != policy.frame_chunk_size:
        raise ValueError("execution and policy_server frame_chunk_size must match")
    if config.execution.lingbot_action_per_frame != policy.action_per_frame:
        raise ValueError("execution and policy_server action_per_frame must match")
    if config.execution.exec_rate <= 0:
        raise ValueError("execution.exec_rate must be positive")
    if config.relay.enable_timeout_s <= 0 or config.relay.max_status_age_s <= 0:
        raise ValueError("relay timeouts must be positive")
    if config.cameras.width <= 0 or config.cameras.height <= 0 or config.cameras.fps <= 0:
        raise ValueError("cameras width/height/fps must be positive")
    if config.cameras.max_camera_age_s <= 0:
        raise ValueError("cameras.max_camera_age_s must be positive")
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
    if any(lo >= hi for lo, hi in zip(config.eef.xyz_min, config.eef.xyz_max, strict=True)):
        raise ValueError("eef.xyz_min values must be lower than eef.xyz_max")
    if config.eef.min_quat_norm <= 0:
        raise ValueError("eef.min_quat_norm must be positive")
    if config.servo.gain <= 0:
        raise ValueError("servo.gain must be positive")
    if config.servo.corrections < 0:
        raise ValueError("servo.corrections must be >= 0")
    if config.gripper.stroke_scale_mm == 0:
        raise ValueError("gripper.stroke_scale_mm must be non-zero")
    if config.gripper.stroke_max_mm <= config.gripper.stroke_min_mm:
        raise ValueError("gripper stroke_max_mm must be greater than stroke_min_mm")
    if (
        config.gripper.stroke_min_mm < config.system.gripper.stroke_min_mm
        or config.gripper.stroke_max_mm > config.system.gripper.stroke_max_mm
    ):
        raise ValueError("gripper policy output stroke must stay inside the physical system range")
    if not 0.0 < config.gripper.command_open_threshold < 1.0:
        raise ValueError("gripper.command_open_threshold must be between 0 and 1")
    if not (
        config.system.gripper.stroke_min_mm
        < config.gripper.feedback_open_threshold_mm
        < config.system.gripper.stroke_max_mm
    ):
        raise ValueError("gripper feedback threshold must be inside the physical system range")
    for name, value in config.topics.__dict__.items():
        if not value.startswith("/"):
            raise ValueError(f"topics.{name} must be an absolute ROS topic: {value!r}")


def bridge_argv(config: LingBotConfig) -> list[str]:
    args = [
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--prompt",
        config.server.prompt,
        _bool_flag("step-mode", config.execution.step_mode),
        "--max-model-calls",
        str(config.execution.max_model_calls),
        "--execute-frames",
        str(config.execution.execute_frames),
        _bool_flag("condition-on-ee-state", config.execution.condition_on_ee_state),
        "--lingbot-frame-chunk-size",
        str(config.execution.lingbot_frame_chunk_size),
        "--lingbot-action-per-frame",
        str(config.execution.lingbot_action_per_frame),
        "--exec-rate",
        _num(config.execution.exec_rate),
        _bool_flag("print-actions", config.execution.print_actions),
        "--review-deadband",
        _num(config.execution.review_deadband_m),
        "--cam-width",
        str(config.cameras.width),
        "--cam-height",
        str(config.cameras.height),
        "--cam-fps",
        str(config.cameras.fps),
        "--max-camera-age",
        _num(config.cameras.max_camera_age_s),
        "--cam0-serial",
        config.cameras.front_serial,
        _bool_flag("cam0-auto-exposure", config.cameras.front_auto_exposure),
        "--cam0-exposure",
        str(config.cameras.front_exposure),
        "--cam0-gain",
        str(config.cameras.front_gain),
        _bool_flag("cam0-auto-white-balance", config.cameras.front_auto_white_balance),
        "--cam0-white-balance",
        str(config.cameras.front_white_balance),
        "--cam0-observation-key",
        config.cameras.front_observation_key,
        _bool_flag("cam0-crop-enabled", config.cameras.front_crop is not None),
        "--cam1-device",
        config.cameras.wrist_device,
        "--cam1-backend",
        config.cameras.wrist_backend,
        "--cam1-serial",
        config.cameras.wrist_serial,
        "--cam1-backend-api",
        config.cameras.wrist_backend_api,
        "--cam1-observation-key",
        config.cameras.wrist_observation_key,
        *web_preview_argv(config.web_preview),
        "--state-pose-topic",
        config.topics.state_pose,
        "--state-gripper-topic",
        config.topics.state_gripper,
        "--cmd-pose-topic",
        config.topics.cmd_pose,
        "--cmd-gripper-topic",
        config.topics.cmd_gripper,
        "--motion-enable-topic",
        config.topics.motion_enable,
        "--relay-status-topic",
        config.topics.relay_status,
        "--relay-enable-timeout",
        _num(config.relay.enable_timeout_s),
        "--max-relay-status-age",
        _num(config.relay.max_status_age_s),
        "--command-frame",
        config.eef.command_frame,
        "--action-pose-mode",
        config.eef.action_pose_mode,
        "--orientation-mode",
        config.eef.orientation_mode,
        "--eef-servo-gain",
        _num(config.servo.gain),
        "--eef-servo-max-extra",
        _num(config.servo.max_extra_m),
        "--eef-servo-settle",
        _num(config.servo.settle_s),
        "--eef-servo-tolerance",
        _num(config.servo.tolerance_m),
        "--eef-servo-corrections",
        str(config.servo.corrections),
        _bool_flag("cache-actual-feedback", config.servo.cache_actual_feedback),
        "--xyz-min",
        *(_num(value) for value in config.eef.xyz_min),
        "--xyz-max",
        *(_num(value) for value in config.eef.xyz_max),
        "--min-quat-norm",
        _num(config.eef.min_quat_norm),
        "--max-feedback-age",
        _num(config.eef.max_feedback_age_s),
        "--feedback-wait-timeout",
        _num(config.eef.feedback_wait_timeout_s),
        "--gripper-stroke-scale",
        _num(config.gripper.stroke_scale_mm),
        "--gripper-stroke-offset",
        _num(config.gripper.stroke_offset_mm),
        "--gripper-stroke-min",
        _num(config.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        _num(config.gripper.stroke_max_mm),
        "--gripper-command-mode",
        config.gripper.command_mode,
        "--gripper-command-open-threshold",
        _num(config.gripper.command_open_threshold),
        "--gripper-feedback-open-threshold-mm",
        _num(config.gripper.feedback_open_threshold_mm),
    ]
    if config.execution.execute:
        args.append("--execute")
    if config.execution.step_actions:
        args.append("--step-actions")
    if config.execution.no_kv_update:
        args.append("--no-kv-update")
    if config.execution.initial_ee_pose is not None:
        args.extend(["--initial-ee-pose", *(_num(value) for value in config.execution.initial_ee_pose)])
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


def bash_config(config: LingBotConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SYSTEM_CONFIG_PATH", str(config.system.path)),
        _assign("SESSION", config.session.tmux),
        _assign("LINGBOT_HOST", config.server.host),
        _assign("LINGBOT_PORT", str(config.server.port)),
        _assign("PROMPT", config.server.prompt),
        _assign("MODEL_SESSION", config.policy_server.tmux),
        _assign("MODEL_CHECKOUT", str(config.policy_server.checkout)),
        _assign("MODEL_PYTHON", str(config.policy_server.python)),
        _assign("BASE_MODEL", str(config.policy_server.base_model)),
        _assign("MODEL_CHECKPOINT", str(config.policy_server.checkpoint)),
        _assign("MODEL_ROOT", str(config.policy_server.model_root)),
        _assign("MODEL_SAVE_ROOT", str(config.policy_server.save_root)),
        _assign("MODEL_MASTER_PORT", str(config.policy_server.master_port)),
        _assign("MODEL_STARTUP_TIMEOUT", _num(config.policy_server.startup_timeout_s)),
        _assign("MODEL_EXPECTED_WEIGHT_SIZE", str(config.policy_server.expected_weight_size_bytes)),
        _assign(
            "DEPLOYMENT_READY",
            "1" if config.policy_server.deployment_ready else "0",
        ),
        _assign("WRIST_BACKEND", config.cameras.wrist_backend),
        _assign("WRIST_SERIAL", config.cameras.wrist_serial),
        _assign("WRIST_CAMERA", config.cameras.wrist_device),
        _assign("STAGED_TOPIC", config.topics.staged_command),
        _assign("RELAY_ENABLE_TOPIC", config.topics.motion_enable),
        _assign("RELAY_STATUS_TOPIC", config.topics.relay_status),
        _array("BRIDGE_ARGS", bridge_argv(config)),
    ]
    return "\n".join(lines)


def _orientation_mode(value: str) -> OrientationMode:
    if value not in ("hold-current", "model-quat"):
        raise ValueError(f"unsupported eef.orientation_mode: {value!r}")
    return value


def _pose_mode(value: str) -> PoseMode:
    if value not in ("absolute", "episode-relative"):
        raise ValueError(f"unsupported eef.action_pose_mode: {value!r}")
    return value


def _gripper_command_mode(value: str) -> GripperCommandMode:
    if value not in ("binary", "continuous"):
        raise ValueError(f"unsupported gripper.command_mode: {value!r}")
    return value


def _text_encoder_device(value: str) -> TextEncoderDevice:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"unsupported policy_server.text_encoder_device: {value!r}")
    return value


def _float_tuple_any(data: dict[str, Any], key: str) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty number list")
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
    parser.add_argument("--shell", action="store_true", help="Emit bash assignments for a1_lingbot_runtime.sh")
    args = parser.parse_args(argv)

    config = load_lingbot_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
