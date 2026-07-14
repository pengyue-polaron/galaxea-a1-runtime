"""Git-tracked teleoperation runtime configuration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.configuration.base import (
    bool_flag as _bool_flag,
    float_tuple as _float_tuple,
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
from galaxea_a1_runtime.teleop.joint_mapping import JointMappingConfig

DEFAULT_TELEOP_CONFIG = Path("configs/teleop/a1_so100.toml")


@dataclass(frozen=True)
class TeleopHostConfig:
    image: str
    a1_serial: str
    prefix: str
    run_dir: str


@dataclass(frozen=True)
class TeleopLeaderConfig:
    port: str
    id: str
    use_degrees: bool


@dataclass(frozen=True)
class TeleopTopicsConfig:
    joint_states: str
    target: str
    staged_command: str
    relay_enable: str
    relay_status: str
    gripper_command: str
    gripper_feedback: str
    eef: str
    host_command: str


@dataclass(frozen=True)
class TeleopBridgeConfig:
    hz: float
    dof: int
    target_joint_names: tuple[str, ...]
    mapping: JointMappingConfig
    relay_enable_timeout_s: float
    max_relay_status_age_s: float
    a1_state_timeout_s: float
    initial_alignment_tolerance_rad: float


@dataclass(frozen=True)
class TeleopGripperConfig:
    enabled: bool
    source_key: str
    min_stroke_mm: float
    max_stroke_mm: float
    invert: bool
    binary_open_threshold: float


@dataclass(frozen=True)
class TeleopCollectionConfig:
    data_root: Path
    state_mode: StateMode
    fps: float
    max_duration_s: float
    auto_reset_after_save: bool
    jpeg_quality: int
    ready_timeout_s: float
    max_camera_age_s: float
    max_joint_action_step_rad: float


@dataclass(frozen=True)
class TeleopCameraConfig:
    width: int
    height: int
    fps: int
    backend: str = "realsense"
    serial: str = ""
    device: str = ""
    pixel_format: str = ""
    require_usb3: bool = False
    depth: bool = False
    depth_width: int = 0
    depth_height: int = 0
    align_depth_to_color: bool = True
    crop: ImageRoi | None = None


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    system: SystemConfig
    host: TeleopHostConfig
    leader: TeleopLeaderConfig
    topics: TeleopTopicsConfig
    bridge: TeleopBridgeConfig
    gripper: TeleopGripperConfig
    collection: TeleopCollectionConfig
    front_camera: TeleopCameraConfig
    wrist_camera: TeleopCameraConfig
    web_preview: WebPreviewConfig


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_TELEOP_CONFIG


def load_teleop_config(path: Path, *, repo_root: Path | None = None) -> TeleopConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    runtime = _required_table(data, "runtime")
    leader = _required_table(data, "leader")
    bridge = _required_table(data, "bridge")
    gripper = _required_table(data, "gripper")
    collection = _required_table(data, "collection")
    front = system.cameras.front
    wrist = system.cameras.wrist

    dof = int(bridge.get("dof", 6))
    mapping = JointMappingConfig(
        relative=bool(bridge.get("relative", True)),
        input_degrees=bool(bridge.get("input_degrees", True)),
        scale=_float_tuple(bridge, "scale", dof),
        sign=_float_tuple(bridge, "sign", dof),
        bias_rad=_float_tuple(bridge, "bias_rad", dof),
        lower_limits=system.joint_safety.lower_limits,
        upper_limits=system.joint_safety.upper_limits,
    )
    mapping.validate(dof)

    config = TeleopConfig(
        path=path,
        system=system,
        host=TeleopHostConfig(
            image=system.host.image,
            a1_serial=system.host.a1_serial,
            prefix=_string(runtime, "prefix"),
            run_dir=_string(runtime, "run_dir"),
        ),
        leader=TeleopLeaderConfig(
            port=_string(leader, "port"),
            id=_string(leader, "id"),
            use_degrees=bool(leader.get("use_degrees", True)),
        ),
        topics=TeleopTopicsConfig(
            joint_states=system.topics.joint_states,
            target=system.topics.joint_target,
            staged_command=system.topics.staged_command,
            relay_enable=system.topics.motion_enable,
            relay_status=system.topics.relay_status,
            gripper_command=system.topics.gripper_command,
            gripper_feedback=system.topics.gripper_feedback,
            eef=system.topics.eef_pose,
            host_command=system.topics.host_command,
        ),
        bridge=TeleopBridgeConfig(
            hz=float(bridge.get("hz", 60.0)),
            dof=dof,
            target_joint_names=system.joint_safety.names,
            mapping=mapping,
            relay_enable_timeout_s=system.relay.enable_timeout_s,
            max_relay_status_age_s=system.relay.max_status_age_s,
            a1_state_timeout_s=float(bridge.get("a1_state_timeout_s", system.joint_safety.state_timeout_s)),
            initial_alignment_tolerance_rad=system.joint_safety.initial_alignment_tolerance_rad,
        ),
        gripper=TeleopGripperConfig(
            enabled=bool(gripper.get("enabled", True)),
            source_key=_string(gripper, "source_key"),
            min_stroke_mm=system.gripper.stroke_min_mm,
            max_stroke_mm=system.gripper.stroke_max_mm,
            invert=bool(gripper.get("invert", False)),
            binary_open_threshold=float(gripper.get("binary_open_threshold", 0.15)),
        ),
        collection=TeleopCollectionConfig(
            data_root=_repo_path(repo_root, _string(collection, "data_root")),
            state_mode=StateMode(_string(collection, "state_mode")),
            fps=float(collection.get("fps", 30.0)),
            max_duration_s=float(collection.get("max_duration_s", 0.0)),
            auto_reset_after_save=bool(collection.get("auto_reset_after_save", True)),
            jpeg_quality=int(collection.get("jpeg_quality", 95)),
            ready_timeout_s=float(collection.get("ready_timeout_s", 10.0)),
            max_camera_age_s=system.cameras.max_age_s,
            max_joint_action_step_rad=float(collection.get("max_joint_action_step_rad", 0.35)),
        ),
        front_camera=TeleopCameraConfig(
            backend=front.backend,
            serial=front.serial,
            device=front.device,
            pixel_format=front.pixel_format,
            width=front.width,
            height=front.height,
            fps=front.fps,
            require_usb3=front.require_usb3,
            depth=front.depth,
            depth_width=front.depth_width,
            depth_height=front.depth_height,
            align_depth_to_color=front.align_depth_to_color,
            crop=front.crop,
        ),
        wrist_camera=TeleopCameraConfig(
            backend=wrist.backend,
            serial=wrist.serial,
            device=wrist.device,
            pixel_format=wrist.pixel_format,
            width=wrist.width,
            height=wrist.height,
            fps=wrist.fps,
        ),
        web_preview=system.web_preview,
    )
    validate_teleop_config(config)
    return config


def validate_teleop_config(config: TeleopConfig) -> None:
    if config.bridge.hz <= 0:
        raise ValueError("bridge.hz must be positive")
    if config.collection.fps <= 0:
        raise ValueError("collection.fps must be positive")
    if config.collection.max_camera_age_s <= 0:
        raise ValueError("collection.max_camera_age_s must be positive")
    if config.collection.max_joint_action_step_rad <= 0:
        raise ValueError("collection.max_joint_action_step_rad must be positive")
    if config.bridge.initial_alignment_tolerance_rad < 0:
        raise ValueError("bridge.initial_alignment_tolerance_rad must be non-negative")
    if config.gripper.max_stroke_mm <= config.gripper.min_stroke_mm:
        raise ValueError("gripper.max_stroke_mm must be greater than min_stroke_mm")
    if not 0.0 < config.gripper.binary_open_threshold < 1.0:
        raise ValueError("gripper.binary_open_threshold must be between 0 and 1")
    for label, camera in (("front", config.front_camera), ("wrist", config.wrist_camera)):
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError(f"cameras.{label} width/height/fps must be positive")
        if camera.depth and (camera.depth_width <= 0 or camera.depth_height <= 0):
            raise ValueError(f"cameras.{label} depth_width/depth_height must be positive")
        if camera.crop is not None:
            camera.crop.validate(
                image_width=camera.width,
                image_height=camera.height,
                label=f"cameras.{label} crop",
            )
        if camera.backend not in {"realsense", "v4l2"}:
            raise ValueError(f"cameras.{label}.backend must be 'realsense' or 'v4l2'")
        if camera.backend == "realsense" and not camera.serial:
            raise ValueError(f"cameras.{label}.serial is required for the RealSense backend")
        if camera.backend == "v4l2" and not camera.device:
            raise ValueError(f"cameras.{label}.device is required for the V4L2 backend")
    if config.front_camera.backend != "realsense":
        raise ValueError("cameras.front.backend must be 'realsense' because teleop records optional depth framesets")
    if config.front_camera.depth and config.front_camera.crop is not None and not config.front_camera.align_depth_to_color:
        raise ValueError("front depth must be aligned to color when cameras.front crop is enabled")
    for name, value in config.topics.__dict__.items():
        if not value.startswith("/"):
            raise ValueError(f"topics.{name} must be an absolute ROS topic: {value!r}")
    if len(config.bridge.target_joint_names) != config.bridge.dof:
        raise ValueError("bridge.target_joint_names length must match bridge.dof")


def bridge_argv(config: TeleopConfig) -> list[str]:
    mapping = config.bridge.mapping
    args = [
        "--leader-port",
        config.leader.port,
        "--leader-id",
        config.leader.id,
        _bool_flag("leader-use-degrees", config.leader.use_degrees),
        "--hz",
        _num(config.bridge.hz),
        "--dof",
        str(config.bridge.dof),
        "--joint-states-topic",
        config.topics.joint_states,
        "--target-topic",
        config.topics.target,
        "--staged-command-topic",
        config.topics.staged_command,
        "--target-joint-names",
        _csv(config.bridge.target_joint_names),
        _bool_flag("relative", mapping.relative),
        _bool_flag("input-degrees", mapping.input_degrees),
        f"--scale={_csv(mapping.scale)}",
        f"--sign={_csv(mapping.sign)}",
        f"--bias-rad={_csv(mapping.bias_rad)}",
        f"--lower-limits={_csv(mapping.lower_limits)}",
        f"--upper-limits={_csv(mapping.upper_limits)}",
        "--motion-enable-topic",
        config.topics.relay_enable,
        "--relay-status-topic",
        config.topics.relay_status,
        "--relay-enable-timeout",
        _num(config.bridge.relay_enable_timeout_s),
        "--max-relay-status-age",
        _num(config.bridge.max_relay_status_age_s),
        "--a1-state-timeout",
        _num(config.bridge.a1_state_timeout_s),
        "--initial-alignment-tolerance",
        _num(config.bridge.initial_alignment_tolerance_rad),
        _bool_flag("gripper-enabled", config.gripper.enabled),
        "--gripper-source-key",
        config.gripper.source_key,
        "--gripper-topic",
        config.topics.gripper_command,
        "--gripper-min-stroke-mm",
        _num(config.gripper.min_stroke_mm),
        "--gripper-max-stroke-mm",
        _num(config.gripper.max_stroke_mm),
        "--gripper-binary-open-threshold",
        _num(config.gripper.binary_open_threshold),
    ]
    if config.gripper.invert:
        args.append("--gripper-invert")
    return args


def collect_argv(config: TeleopConfig) -> list[str]:
    args = [
        "--data-root",
        str(config.collection.data_root),
        "--state-mode",
        config.collection.state_mode.value,
        "--fps",
        _num(config.collection.fps),
        "--max-duration-s",
        _num(config.collection.max_duration_s),
        _bool_flag("auto-reset-after-save", config.collection.auto_reset_after_save),
        "--jpeg-quality",
        str(config.collection.jpeg_quality),
        "--ready-timeout-s",
        _num(config.collection.ready_timeout_s),
        "--max-camera-age-s",
        _num(config.collection.max_camera_age_s),
        "--max-joint-action-step-rad",
        _num(config.collection.max_joint_action_step_rad),
        "--joint-topic",
        config.topics.joint_states,
        "--eef-topic",
        config.topics.eef,
        "--action-topic",
        config.topics.target,
        "--gripper-feedback-topic",
        config.topics.gripper_feedback,
        "--gripper-action-topic",
        config.topics.gripper_command,
        "--gripper-stroke-scale",
        _num(config.gripper.max_stroke_mm),
        "--gripper-binary-open-threshold",
        _num(config.gripper.binary_open_threshold),
        "--staged-command-topic",
        config.topics.staged_command,
        "--host-command-topic",
        config.topics.host_command,
        "--cam0-width",
        str(config.front_camera.width),
        "--cam0-height",
        str(config.front_camera.height),
        "--cam0-fps",
        str(config.front_camera.fps),
        _bool_flag("cam0-require-usb3", config.front_camera.require_usb3),
        _bool_flag("cam0-depth-enabled", config.front_camera.depth),
        "--cam0-depth-width",
        str(config.front_camera.depth_width or config.front_camera.width),
        "--cam0-depth-height",
        str(config.front_camera.depth_height or config.front_camera.height),
        _bool_flag("cam0-align-depth-to-color", config.front_camera.align_depth_to_color),
        _bool_flag("cam0-crop-enabled", config.front_camera.crop is not None),
        "--cam1-device",
        config.wrist_camera.device,
        "--cam1-backend",
        config.wrist_camera.backend,
        "--cam1-serial",
        config.wrist_camera.serial,
        "--cam1-width",
        str(config.wrist_camera.width),
        "--cam1-height",
        str(config.wrist_camera.height),
        "--cam1-fps",
        str(config.wrist_camera.fps),
        "--cam1-pixel-format",
        config.wrist_camera.pixel_format,
        *web_preview_argv(config.web_preview),
    ]
    if config.front_camera.crop is not None:
        args.extend(
            [
                "--cam0-crop-x",
                str(config.front_camera.crop.x),
                "--cam0-crop-y",
                str(config.front_camera.crop.y),
                "--cam0-crop-width",
                str(config.front_camera.crop.width),
                "--cam0-crop-height",
                str(config.front_camera.crop.height),
            ]
        )
    args.extend(["--cam0-serial", config.front_camera.serial])
    return args


def bash_config(config: TeleopConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SYSTEM_CONFIG_PATH", str(config.system.path)),
        _assign("IMAGE", config.host.image),
        _assign("SERIAL", config.host.a1_serial),
        _assign("LEADER_PORT", config.leader.port),
        _assign("LEADER_ID", config.leader.id),
        _assign("PREFIX", config.host.prefix),
        _assign("RUN_DIR", config.host.run_dir),
        _assign("STAGED_TOPIC", config.topics.staged_command),
        _assign("RELAY_ENABLE_TOPIC", config.topics.relay_enable),
        _assign("RELAY_STATUS_TOPIC", config.topics.relay_status),
        _assign("TARGET_TOPIC", config.topics.target),
        _assign("GRIPPER_MIN_STROKE_MM", _num(config.gripper.min_stroke_mm)),
        _assign("GRIPPER_MAX_STROKE_MM", _num(config.gripper.max_stroke_mm)),
        _assign("WEB_PREVIEW_PORT", str(config.web_preview.port)),
        _array("BRIDGE_ARGS", bridge_argv(config)),
        _array("COLLECT_ARGS", collect_argv(config)),
    ]
    return "\n".join(lines)


def _csv(values: tuple[float, ...] | tuple[str, ...]) -> str:
    return ",".join(_num(value) if isinstance(value, float) else str(value) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read A1 teleop TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shell", action="store_true", help="Emit bash assignments for a1_teleop_runtime.sh")
    args = parser.parse_args(argv)

    config = load_teleop_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
