"""Typed configuration for the physical A1 system shared by every app."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple,
    floating,
    integer,
    integer_tuple,
    load_toml,
    number,
    require_exact_keys,
    required_table,
    shell_assign,
    string,
    string_tuple,
)
from galaxea_a1_runtime.configuration.cameras import (
    SystemCameraDeviceConfig,
    SystemCamerasConfig,
    SystemRealSenseCameraConfig,
    SystemV4l2CameraConfig,
    parse_system_cameras,
)
from galaxea_a1_runtime.configuration.camera_diagnostics import (
    CameraDiagnosticsConfig,
    parse_camera_diagnostics_config,
)
from galaxea_a1_runtime.configuration.cli import run_config_renderer
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.web_preview import (
    WebPreviewConfig,
    parse_web_preview_config,
)
from galaxea_a1_runtime.constants import EE_TRACKER_NODE, JOINT_TRACKER_NODE
from galaxea_a1_runtime.constants import ARM_JOINT_COUNT

__all__ = [
    "DEFAULT_SYSTEM_CONFIG",
    "SystemCameraDeviceConfig",
    "CameraDiagnosticsConfig",
    "SystemCamerasConfig",
    "SystemConfig",
    "SystemDoctorConfig",
    "SystemEefConfig",
    "SystemEefTestConfig",
    "SystemGripperConfig",
    "SystemHostConfig",
    "SystemJointSafetyConfig",
    "SystemRealSenseCameraConfig",
    "SystemRelayConfig",
    "SystemStartupConfig",
    "SystemTopicsConfig",
    "SystemV4l2CameraConfig",
    "bash_config",
    "load_system_config",
    "render_shell_values",
]

DEFAULT_SYSTEM_CONFIG = SYSTEM_CONFIG
ROS_ABSOLUTE_NAME = re.compile(
    r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$"
)


@dataclass(frozen=True)
class SystemHostConfig:
    image: str
    a1_serial: str


@dataclass(frozen=True)
class SystemTopicsConfig:
    joint_states: str
    joint_target: str
    staged_command: str
    host_command: str
    motor_status: str
    motion_enable: str
    relay_status: str
    gripper_target: str
    gripper_command: str
    gripper_feedback: str
    eef_pose: str
    eef_target: str


@dataclass(frozen=True)
class SystemRelayConfig:
    rate_hz: float
    status_rate_hz: float
    enable_timeout_s: float
    max_status_age_s: float
    max_input_age_s: float
    arming_timeout_s: float
    allowed_control_modes: tuple[int, ...]


@dataclass(frozen=True)
class SystemDoctorConfig:
    ros_topic_timeout_s: float


@dataclass(frozen=True)
class SystemStartupConfig:
    ros_master_timeout_s: float
    joint_feedback_timeout_s: float
    topic_timeout_s: float
    tmux_process_grace_s: int


@dataclass(frozen=True)
class SystemJointSafetyConfig:
    names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    action_step_guard_enabled: bool
    max_action_step_rad: float
    max_first_target_delta_rad: float
    initial_alignment_tolerance_rad: float
    state_timeout_s: float
    max_feedback_age_s: float


@dataclass(frozen=True)
class SystemEefConfig:
    command_frame: str
    orientation_mode: str
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    min_quat_norm: float
    max_feedback_age_s: float
    feedback_wait_timeout_s: float


@dataclass(frozen=True)
class SystemEefTestConfig:
    step_m: float
    settle_s: float


@dataclass(frozen=True)
class SystemGripperConfig:
    stroke_min_mm: float
    stroke_max_mm: float


@dataclass(frozen=True)
class SystemConfig:
    path: Path
    host: SystemHostConfig
    topics: SystemTopicsConfig
    relay: SystemRelayConfig
    doctor: SystemDoctorConfig
    startup: SystemStartupConfig
    joint_safety: SystemJointSafetyConfig
    eef: SystemEefConfig
    eef_test: SystemEefTestConfig
    gripper: SystemGripperConfig
    cameras: SystemCamerasConfig
    camera_diagnostics: CameraDiagnosticsConfig
    web_preview: WebPreviewConfig


def load_system_config(path: Path, *, repo_root: Path | None = None) -> SystemConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "host",
            "topics",
            "relay",
            "doctor",
            "startup",
            "joint_safety",
            "eef",
            "eef_test",
            "gripper",
            "cameras",
            "camera_diagnostics",
            "web_preview",
        },
        label="system config",
    )
    host = required_table(data, "host")
    topics = required_table(data, "topics")
    relay = required_table(data, "relay")
    doctor = required_table(data, "doctor")
    startup = required_table(data, "startup")
    joint = required_table(data, "joint_safety")
    eef = required_table(data, "eef")
    eef_test = required_table(data, "eef_test")
    gripper = required_table(data, "gripper")
    cameras = required_table(data, "cameras")
    require_exact_keys(host, required={"image", "a1_serial"}, label="host")
    require_exact_keys(
        topics, required=set(SystemTopicsConfig.__annotations__), label="topics"
    )
    require_exact_keys(
        relay, required=set(SystemRelayConfig.__annotations__), label="relay"
    )
    require_exact_keys(
        doctor, required=set(SystemDoctorConfig.__annotations__), label="doctor"
    )
    require_exact_keys(
        startup, required=set(SystemStartupConfig.__annotations__), label="startup"
    )
    require_exact_keys(
        joint,
        required=set(SystemJointSafetyConfig.__annotations__),
        label="joint_safety",
    )
    require_exact_keys(eef, required=set(SystemEefConfig.__annotations__), label="eef")
    require_exact_keys(
        eef_test, required=set(SystemEefTestConfig.__annotations__), label="eef_test"
    )
    require_exact_keys(
        gripper, required=set(SystemGripperConfig.__annotations__), label="gripper"
    )
    config = SystemConfig(
        path=path,
        host=SystemHostConfig(
            image=string(host, "image"), a1_serial=string(host, "a1_serial")
        ),
        topics=SystemTopicsConfig(
            joint_states=string(topics, "joint_states"),
            joint_target=string(topics, "joint_target"),
            staged_command=string(topics, "staged_command"),
            host_command=string(topics, "host_command"),
            motor_status=string(topics, "motor_status"),
            motion_enable=string(topics, "motion_enable"),
            relay_status=string(topics, "relay_status"),
            gripper_target=string(topics, "gripper_target"),
            gripper_command=string(topics, "gripper_command"),
            gripper_feedback=string(topics, "gripper_feedback"),
            eef_pose=string(topics, "eef_pose"),
            eef_target=string(topics, "eef_target"),
        ),
        relay=SystemRelayConfig(
            rate_hz=floating(relay, "rate_hz"),
            status_rate_hz=floating(relay, "status_rate_hz"),
            enable_timeout_s=floating(relay, "enable_timeout_s"),
            max_status_age_s=floating(relay, "max_status_age_s"),
            max_input_age_s=floating(relay, "max_input_age_s"),
            arming_timeout_s=floating(relay, "arming_timeout_s"),
            allowed_control_modes=integer_tuple(relay, "allowed_control_modes"),
        ),
        doctor=SystemDoctorConfig(
            ros_topic_timeout_s=floating(doctor, "ros_topic_timeout_s"),
        ),
        startup=SystemStartupConfig(
            ros_master_timeout_s=floating(startup, "ros_master_timeout_s"),
            joint_feedback_timeout_s=floating(startup, "joint_feedback_timeout_s"),
            topic_timeout_s=floating(startup, "topic_timeout_s"),
            tmux_process_grace_s=integer(startup, "tmux_process_grace_s"),
        ),
        joint_safety=SystemJointSafetyConfig(
            names=string_tuple(joint, "names", ARM_JOINT_COUNT),
            lower_limits=float_tuple(joint, "lower_limits", ARM_JOINT_COUNT),
            upper_limits=float_tuple(joint, "upper_limits", ARM_JOINT_COUNT),
            action_step_guard_enabled=boolean(joint, "action_step_guard_enabled"),
            max_action_step_rad=floating(joint, "max_action_step_rad"),
            max_first_target_delta_rad=floating(joint, "max_first_target_delta_rad"),
            initial_alignment_tolerance_rad=floating(
                joint, "initial_alignment_tolerance_rad"
            ),
            state_timeout_s=floating(joint, "state_timeout_s"),
            max_feedback_age_s=floating(joint, "max_feedback_age_s"),
        ),
        eef=SystemEefConfig(
            command_frame=string(eef, "command_frame"),
            orientation_mode=string(eef, "orientation_mode"),
            xyz_min=float_tuple(eef, "xyz_min", 3),
            xyz_max=float_tuple(eef, "xyz_max", 3),
            min_quat_norm=floating(eef, "min_quat_norm"),
            max_feedback_age_s=floating(eef, "max_feedback_age_s"),
            feedback_wait_timeout_s=floating(eef, "feedback_wait_timeout_s"),
        ),
        eef_test=SystemEefTestConfig(
            step_m=floating(eef_test, "step_m"),
            settle_s=floating(eef_test, "settle_s"),
        ),
        gripper=SystemGripperConfig(
            stroke_min_mm=floating(gripper, "stroke_min_mm"),
            stroke_max_mm=floating(gripper, "stroke_max_mm"),
        ),
        cameras=parse_system_cameras(cameras),
        camera_diagnostics=parse_camera_diagnostics_config(
            required_table(data, "camera_diagnostics"), repo_root=repo_root
        ),
        web_preview=parse_web_preview_config(
            required_table(data, "web_preview"), repo_root=repo_root
        ),
    )
    validate_system_config(config)
    return config


def validate_system_config(config: SystemConfig) -> None:
    if not config.host.a1_serial.startswith("/dev/") or any(
        character.isspace() for character in config.host.a1_serial
    ):
        raise ValueError("host.a1_serial must be a whitespace-free path under /dev")
    for name, value in config.topics.__dict__.items():
        if ROS_ABSOLUTE_NAME.fullmatch(value) is None:
            raise ValueError(
                f"topics.{name} must be a valid absolute ROS name: {value!r}"
            )
    if config.topics.staged_command == config.topics.host_command:
        raise ValueError("topics.staged_command must differ from topics.host_command")
    if config.topics.gripper_target == config.topics.gripper_command:
        raise ValueError(
            "topics.gripper_target must differ from topics.gripper_command"
        )
    if any(
        lo >= hi
        for lo, hi in zip(
            config.joint_safety.lower_limits,
            config.joint_safety.upper_limits,
            strict=True,
        )
    ):
        raise ValueError("joint_safety lower_limits must be below upper_limits")
    if len(set(config.joint_safety.names)) != len(config.joint_safety.names):
        raise ValueError("joint_safety.names must not contain duplicates")
    for name, value in (
        ("max_action_step_rad", config.joint_safety.max_action_step_rad),
        ("max_first_target_delta_rad", config.joint_safety.max_first_target_delta_rad),
        (
            "initial_alignment_tolerance_rad",
            config.joint_safety.initial_alignment_tolerance_rad,
        ),
        ("state_timeout_s", config.joint_safety.state_timeout_s),
        ("max_feedback_age_s", config.joint_safety.max_feedback_age_s),
    ):
        if value <= 0:
            raise ValueError(f"joint_safety.{name} must be positive")
    if config.eef.orientation_mode not in {"hold-current", "model-quat"}:
        raise ValueError("eef.orientation_mode must be hold-current or model-quat")
    if any(
        lo >= hi for lo, hi in zip(config.eef.xyz_min, config.eef.xyz_max, strict=True)
    ):
        raise ValueError("eef.xyz_min must be below xyz_max")
    if (
        min(
            config.eef.min_quat_norm,
            config.eef.max_feedback_age_s,
            config.eef.feedback_wait_timeout_s,
        )
        <= 0
    ):
        raise ValueError("eef quaternion and feedback limits must be positive")
    if config.eef_test.step_m <= 0 or config.eef_test.settle_s < 0:
        raise ValueError("eef_test step must be positive and settle time non-negative")
    if config.gripper.stroke_max_mm <= config.gripper.stroke_min_mm:
        raise ValueError("gripper stroke range is invalid")
    if (
        min(
            config.relay.enable_timeout_s,
            config.relay.rate_hz,
            config.relay.status_rate_hz,
            config.relay.max_status_age_s,
            config.relay.max_input_age_s,
            config.relay.arming_timeout_s,
        )
        <= 0
    ):
        raise ValueError("relay timeouts must be positive")
    if not config.relay.allowed_control_modes:
        raise ValueError("relay.allowed_control_modes must not be empty")
    if len(set(config.relay.allowed_control_modes)) != len(
        config.relay.allowed_control_modes
    ) or any(mode < 0 or mode > 255 for mode in config.relay.allowed_control_modes):
        raise ValueError("relay.allowed_control_modes must contain unique uint8 values")
    if config.doctor.ros_topic_timeout_s <= 0:
        raise ValueError("doctor.ros_topic_timeout_s must be positive")
    if (
        min(
            config.startup.ros_master_timeout_s,
            config.startup.joint_feedback_timeout_s,
            config.startup.topic_timeout_s,
            config.startup.tmux_process_grace_s,
        )
        < 1
    ):
        raise ValueError("startup timeouts must be at least one second")


def shell_values(config: SystemConfig) -> dict[str, str]:
    """Return the canonical system-to-shell lifecycle mapping."""

    return {
        "SYSTEM_CONFIG_PATH": str(config.path),
        "IMAGE": config.host.image,
        "SERIAL": config.host.a1_serial,
        "JOINT_STATES_TOPIC": config.topics.joint_states,
        "JOINT_TARGET_TOPIC": config.topics.joint_target,
        "STAGED_TOPIC": config.topics.staged_command,
        "RELAY_STATUS_TOPIC": config.topics.relay_status,
        "EEF_POSE_TOPIC": config.topics.eef_pose,
        "EEF_TARGET_TOPIC": config.topics.eef_target,
        "EE_TRACKER_NODE": EE_TRACKER_NODE,
        "JOINT_TRACKER_NODE": JOINT_TRACKER_NODE,
        "WRIST_BACKEND": config.cameras.wrist.backend,
        "WRIST_CAMERA": (
            config.cameras.wrist.device
            if isinstance(config.cameras.wrist, SystemV4l2CameraConfig)
            else ""
        ),
        "GRIPPER_MIN_STROKE_MM": number(config.gripper.stroke_min_mm),
        "GRIPPER_MAX_STROKE_MM": number(config.gripper.stroke_max_mm),
        "ROS_MASTER_STARTUP_TIMEOUT_S": number(config.startup.ros_master_timeout_s),
        "JOINT_FEEDBACK_STARTUP_TIMEOUT_S": number(
            config.startup.joint_feedback_timeout_s
        ),
        "TOPIC_STARTUP_TIMEOUT_S": number(config.startup.topic_timeout_s),
        "TMUX_STARTUP_GRACE_S": str(config.startup.tmux_process_grace_s),
    }


def render_shell_values(config: SystemConfig, names: Iterable[str]) -> str:
    values = shell_values(config)
    requested = tuple(names)
    unknown = tuple(name for name in requested if name not in values)
    if unknown:
        raise ValueError(f"unknown system shell value(s): {list(unknown)}")
    return "\n".join(shell_assign(name, values[name]) for name in requested)


def bash_config(config: SystemConfig) -> str:
    """Emit the physical runtime contract for the boring shell entrypoints."""

    return render_shell_values(
        config,
        (
            "SYSTEM_CONFIG_PATH",
            "IMAGE",
            "SERIAL",
            "JOINT_STATES_TOPIC",
            "JOINT_TARGET_TOPIC",
            "STAGED_TOPIC",
            "RELAY_STATUS_TOPIC",
            "EEF_POSE_TOPIC",
            "EEF_TARGET_TOPIC",
            "EE_TRACKER_NODE",
            "JOINT_TRACKER_NODE",
            "ROS_MASTER_STARTUP_TIMEOUT_S",
            "JOINT_FEEDBACK_STARTUP_TIMEOUT_S",
            "TOPIC_STARTUP_TIMEOUT_S",
            "TMUX_STARTUP_GRACE_S",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    return run_config_renderer(
        argv,
        description="Read the unique A1 physical system config.",
        default_config=DEFAULT_SYSTEM_CONFIG,
        load_config=load_system_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
