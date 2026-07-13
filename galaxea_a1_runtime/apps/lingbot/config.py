"""Git-tracked LingBot-VA runtime configuration."""

from __future__ import annotations

import argparse
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OrientationMode = Literal["hold-current", "model-quat"]

DEFAULT_LINGBOT_CONFIG = Path("configs/inference/lingbot_va_a1.toml")


@dataclass(frozen=True)
class LingBotSessionConfig:
    tmux: str


@dataclass(frozen=True)
class LingBotServerConfig:
    host: str
    port: int
    prompt: str


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
    command_open_threshold: float
    feedback_open_threshold_mm: float


@dataclass(frozen=True)
class LingBotCameraConfig:
    width: int
    height: int
    fps: int
    front_serial: str
    front_auto_exposure: bool
    front_exposure: int
    front_gain: int
    front_auto_white_balance: bool
    front_white_balance: int
    wrist_device: str
    wrist_backend_api: str


@dataclass(frozen=True)
class LingBotConfig:
    path: Path
    session: LingBotSessionConfig
    server: LingBotServerConfig
    execution: LingBotExecutionConfig
    topics: LingBotTopicsConfig
    relay: LingBotRelayConfig
    eef: LingBotEefConfig
    servo: LingBotServoConfig
    gripper: LingBotGripperConfig
    cameras: LingBotCameraConfig


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LINGBOT_CONFIG


def load_lingbot_config(path: Path, *, repo_root: Path | None = None) -> LingBotConfig:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    path = path.resolve()
    data = tomllib.loads(path.read_text())
    repo_root = repo_root.resolve() if repo_root is not None else path.parents[2]
    del repo_root

    session = _required_table(data, "session")
    server = _required_table(data, "server")
    execution = _required_table(data, "execution")
    topics = _required_table(data, "topics")
    relay = _required_table(data, "relay")
    eef = _required_table(data, "eef")
    servo = _required_table(data, "servo")
    gripper = _required_table(data, "gripper")
    cameras = _required_table(data, "cameras")
    front = _required_table(cameras, "front")
    wrist = _required_table(cameras, "wrist")

    config = LingBotConfig(
        path=path,
        session=LingBotSessionConfig(tmux=_string(session, "tmux")),
        server=LingBotServerConfig(
            host=_string(server, "host"),
            port=int(server.get("port", 1106)),
            prompt=_string(server, "prompt"),
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
            state_pose=_string(topics, "state_pose"),
            state_gripper=_string(topics, "state_gripper"),
            cmd_pose=_string(topics, "cmd_pose"),
            cmd_gripper=_string(topics, "cmd_gripper"),
            motion_enable=_string(topics, "motion_enable"),
            relay_status=_string(topics, "relay_status"),
            staged_command=_string(topics, "staged_command"),
        ),
        relay=LingBotRelayConfig(
            enable_timeout_s=float(relay.get("enable_timeout_s", 2.0)),
            max_status_age_s=float(relay.get("max_status_age_s", 1.0)),
        ),
        eef=LingBotEefConfig(
            command_frame=_string(eef, "command_frame"),
            orientation_mode=_orientation_mode(_string(eef, "orientation_mode")),
            xyz_min=_float_tuple(eef, "xyz_min", 3),
            xyz_max=_float_tuple(eef, "xyz_max", 3),
            min_quat_norm=float(eef.get("min_quat_norm", 0.25)),
            max_feedback_age_s=float(eef.get("max_feedback_age_s", 0.5)),
            feedback_wait_timeout_s=float(eef.get("feedback_wait_timeout_s", 5.0)),
        ),
        servo=LingBotServoConfig(
            gain=float(servo.get("gain", 1.0)),
            max_extra_m=float(servo.get("max_extra_m", 0.04)),
            settle_s=float(servo.get("settle_s", 0.0)),
            tolerance_m=float(servo.get("tolerance_m", 0.01)),
            corrections=int(servo.get("corrections", 0)),
            cache_actual_feedback=bool(servo.get("cache_actual_feedback", True)),
        ),
        gripper=LingBotGripperConfig(
            stroke_scale_mm=float(gripper.get("stroke_scale_mm", 200.0)),
            stroke_offset_mm=float(gripper.get("stroke_offset_mm", 0.0)),
            stroke_min_mm=float(gripper.get("stroke_min_mm", 0.0)),
            stroke_max_mm=float(gripper.get("stroke_max_mm", 200.0)),
            command_open_threshold=float(gripper.get("command_open_threshold", 0.5)),
            feedback_open_threshold_mm=float(gripper.get("feedback_open_threshold_mm", 30.0)),
        ),
        cameras=LingBotCameraConfig(
            width=int(cameras.get("width", 640)),
            height=int(cameras.get("height", 480)),
            fps=int(cameras.get("fps", 30)),
            front_serial=str(front.get("serial", "")),
            front_auto_exposure=bool(front.get("auto_exposure", True)),
            front_exposure=int(front.get("exposure", 140)),
            front_gain=int(front.get("gain", 32)),
            front_auto_white_balance=bool(front.get("auto_white_balance", True)),
            front_white_balance=int(front.get("white_balance", 4600)),
            wrist_device=_string(wrist, "device"),
            wrist_backend_api=_string(wrist, "backend_api"),
        ),
    )
    validate_lingbot_config(config)
    return config


def validate_lingbot_config(config: LingBotConfig) -> None:
    if config.server.port <= 0:
        raise ValueError("server.port must be positive")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.execute_frames <= 0:
        raise ValueError("execution.execute_frames must be positive")
    if config.execution.initial_ee_pose is not None and len(config.execution.initial_ee_pose) != 8:
        raise ValueError("execution.initial_ee_pose must contain 8 values")
    if config.execution.lingbot_frame_chunk_size <= 0 or config.execution.lingbot_action_per_frame <= 0:
        raise ValueError("LingBot frame/action dimensions must be positive")
    if config.execution.exec_rate <= 0:
        raise ValueError("execution.exec_rate must be positive")
    if config.relay.enable_timeout_s <= 0 or config.relay.max_status_age_s <= 0:
        raise ValueError("relay timeouts must be positive")
    if config.cameras.width <= 0 or config.cameras.height <= 0 or config.cameras.fps <= 0:
        raise ValueError("cameras width/height/fps must be positive")
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
    if not 0.0 < config.gripper.command_open_threshold < 1.0:
        raise ValueError("gripper.command_open_threshold must be between 0 and 1")
    if not config.gripper.stroke_min_mm < config.gripper.feedback_open_threshold_mm < config.gripper.stroke_max_mm:
        raise ValueError("gripper.feedback_open_threshold_mm must be inside the physical stroke range")
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
        "--cam1-device",
        config.cameras.wrist_device,
        "--cam1-backend-api",
        config.cameras.wrist_backend_api,
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
    return args


def bash_config(config: LingBotConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SESSION", config.session.tmux),
        _assign("LINGBOT_HOST", config.server.host),
        _assign("LINGBOT_PORT", str(config.server.port)),
        _assign("PROMPT", config.server.prompt),
        _assign("WRIST_CAMERA", config.cameras.wrist_device),
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


def _orientation_mode(value: str) -> OrientationMode:
    if value not in ("hold-current", "model-quat"):
        raise ValueError(f"unsupported eef.orientation_mode: {value!r}")
    return value


def _float_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    out = tuple(float(item) for item in value)
    if len(out) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(out)}")
    return out


def _optional_float_tuple(data: dict[str, Any], key: str) -> tuple[float, ...] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    return tuple(float(item) for item in value)


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _num(value: float) -> str:
    return f"{float(value):g}"


def _assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _array(name: str, values: list[str]) -> str:
    return f"{name}=({' '.join(shlex.quote(value) for value in values)})"


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
