"""Typed configuration for the physical A1 system shared by every app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    float_tuple,
    load_toml,
    required_table,
    string,
    string_tuple,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, parse_optional_image_roi
from galaxea_a1_runtime.hardware.web_preview import WebPreviewConfig, parse_web_preview_config


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
    motion_enable: str
    relay_status: str
    gripper_command: str
    gripper_feedback: str
    eef_pose: str
    eef_target: str


@dataclass(frozen=True)
class SystemRelayConfig:
    enable_timeout_s: float
    max_status_age_s: float


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
class SystemGripperConfig:
    stroke_min_mm: float
    stroke_max_mm: float


@dataclass(frozen=True)
class SystemCameraDeviceConfig:
    backend: str
    serial: str
    device: str
    width: int
    height: int
    fps: int
    backend_api: str
    pixel_format: str
    require_usb3: bool
    depth: bool
    depth_width: int
    depth_height: int
    align_depth_to_color: bool
    auto_exposure: bool
    exposure: int
    gain: int
    auto_white_balance: bool
    white_balance: int
    crop: ImageRoi | None


@dataclass(frozen=True)
class SystemCamerasConfig:
    warmup_frames: int
    max_age_s: float
    front: SystemCameraDeviceConfig
    wrist: SystemCameraDeviceConfig


@dataclass(frozen=True)
class SystemConfig:
    path: Path
    host: SystemHostConfig
    topics: SystemTopicsConfig
    relay: SystemRelayConfig
    joint_safety: SystemJointSafetyConfig
    eef: SystemEefConfig
    gripper: SystemGripperConfig
    cameras: SystemCamerasConfig
    web_preview: WebPreviewConfig


def load_system_config(path: Path, *, repo_root: Path | None = None) -> SystemConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    host = required_table(data, "host")
    topics = required_table(data, "topics")
    relay = required_table(data, "relay")
    joint = required_table(data, "joint_safety")
    eef = required_table(data, "eef")
    gripper = required_table(data, "gripper")
    cameras = required_table(data, "cameras")
    front = required_table(cameras, "front")
    wrist = required_table(cameras, "wrist")
    front_width, front_height = int(front.get("width", 640)), int(front.get("height", 480))
    wrist_width, wrist_height = int(wrist.get("width", 640)), int(wrist.get("height", 480))

    config = SystemConfig(
        path=path,
        host=SystemHostConfig(image=string(host, "image"), a1_serial=string(host, "a1_serial")),
        topics=SystemTopicsConfig(
            joint_states=string(topics, "joint_states"),
            joint_target=string(topics, "joint_target"),
            staged_command=string(topics, "staged_command"),
            host_command=string(topics, "host_command"),
            motion_enable=string(topics, "motion_enable"),
            relay_status=string(topics, "relay_status"),
            gripper_command=string(topics, "gripper_command"),
            gripper_feedback=string(topics, "gripper_feedback"),
            eef_pose=string(topics, "eef_pose"),
            eef_target=string(topics, "eef_target"),
        ),
        relay=SystemRelayConfig(
            enable_timeout_s=float(relay.get("enable_timeout_s", 2.0)),
            max_status_age_s=float(relay.get("max_status_age_s", 1.0)),
        ),
        joint_safety=SystemJointSafetyConfig(
            names=string_tuple(joint, "names", 6),
            lower_limits=float_tuple(joint, "lower_limits", 6),
            upper_limits=float_tuple(joint, "upper_limits", 6),
            action_step_guard_enabled=bool(joint.get("action_step_guard_enabled", False)),
            max_action_step_rad=float(joint.get("max_action_step_rad", 0.25)),
            max_first_target_delta_rad=float(joint.get("max_first_target_delta_rad", 0.25)),
            initial_alignment_tolerance_rad=float(joint.get("initial_alignment_tolerance_rad", 0.05)),
            state_timeout_s=float(joint.get("state_timeout_s", 10.0)),
            max_feedback_age_s=float(joint.get("max_feedback_age_s", 0.5)),
        ),
        eef=SystemEefConfig(
            command_frame=string(eef, "command_frame"),
            orientation_mode=string(eef, "orientation_mode"),
            xyz_min=float_tuple(eef, "xyz_min", 3),
            xyz_max=float_tuple(eef, "xyz_max", 3),
            min_quat_norm=float(eef.get("min_quat_norm", 0.25)),
            max_feedback_age_s=float(eef.get("max_feedback_age_s", 0.5)),
            feedback_wait_timeout_s=float(eef.get("feedback_wait_timeout_s", 5.0)),
        ),
        gripper=SystemGripperConfig(
            stroke_min_mm=float(gripper.get("stroke_min_mm", 0.0)),
            stroke_max_mm=float(gripper.get("stroke_max_mm", 200.0)),
        ),
        cameras=SystemCamerasConfig(
            warmup_frames=int(cameras.get("warmup_frames", 20)),
            max_age_s=float(cameras.get("max_age_s", 0.5)),
            front=_camera(front, front_width, front_height, require_square_crop=True),
            wrist=_camera(wrist, wrist_width, wrist_height, require_square_crop=False),
        ),
        web_preview=parse_web_preview_config(
            data.get("web_preview", {}) if isinstance(data.get("web_preview", {}), dict) else {},
            repo_root=repo_root,
        ),
    )
    validate_system_config(config)
    return config


def _camera(data: dict[str, Any], width: int, height: int, *, require_square_crop: bool) -> SystemCameraDeviceConfig:
    return SystemCameraDeviceConfig(
        backend=str(data.get("backend", "realsense")),
        serial=str(data.get("serial", "")),
        device=str(data.get("device", "")),
        width=width,
        height=height,
        fps=int(data.get("fps", 30)),
        backend_api=str(data.get("backend_api", "v4l2")),
        pixel_format=str(data.get("pixel_format", "")),
        require_usb3=bool(data.get("require_usb3", False)),
        depth=bool(data.get("depth", False)),
        depth_width=int(data.get("depth_width", width)),
        depth_height=int(data.get("depth_height", height)),
        align_depth_to_color=bool(data.get("align_depth_to_color", True)),
        auto_exposure=bool(data.get("auto_exposure", True)),
        exposure=int(data.get("exposure", 140)),
        gain=int(data.get("gain", 32)),
        auto_white_balance=bool(data.get("auto_white_balance", True)),
        white_balance=int(data.get("white_balance", 4600)),
        crop=parse_optional_image_roi(
            data,
            image_width=width,
            image_height=height,
            label="system camera crop",
            require_square=require_square_crop,
        ),
    )


def validate_system_config(config: SystemConfig) -> None:
    for name, value in config.topics.__dict__.items():
        if not value.startswith("/"):
            raise ValueError(f"topics.{name} must be absolute: {value!r}")
    if any(lo >= hi for lo, hi in zip(config.joint_safety.lower_limits, config.joint_safety.upper_limits, strict=True)):
        raise ValueError("joint_safety lower_limits must be below upper_limits")
    if config.eef.orientation_mode not in {"hold-current", "model-quat"}:
        raise ValueError("eef.orientation_mode must be hold-current or model-quat")
    if any(lo >= hi for lo, hi in zip(config.eef.xyz_min, config.eef.xyz_max, strict=True)):
        raise ValueError("eef.xyz_min must be below xyz_max")
    if config.gripper.stroke_max_mm <= config.gripper.stroke_min_mm:
        raise ValueError("gripper stroke range is invalid")
    if config.cameras.front.crop is None:
        raise ValueError("system AgentView crop must be enabled")
    for name, camera in (("front", config.cameras.front), ("wrist", config.cameras.wrist)):
        if min(camera.width, camera.height, camera.fps) <= 0:
            raise ValueError(f"cameras.{name} dimensions/fps must be positive")
        if camera.backend == "realsense" and not camera.serial:
            raise ValueError(f"cameras.{name}.serial is required")
