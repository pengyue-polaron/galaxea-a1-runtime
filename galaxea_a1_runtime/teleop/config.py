"""Git-tracked teleoperation runtime configuration."""

from __future__ import annotations

import argparse
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.collection import StateMode
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


@dataclass(frozen=True)
class TeleopCollectionConfig:
    data_root: Path
    state_mode: StateMode
    fps: float
    max_duration_s: float
    jpeg_quality: int
    ready_timeout_s: float


@dataclass(frozen=True)
class TeleopCameraConfig:
    width: int
    height: int
    fps: int
    serial: str = ""
    device: str = ""
    depth: bool = False
    depth_width: int = 0
    depth_height: int = 0
    align_depth_to_color: bool = True


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    host: TeleopHostConfig
    leader: TeleopLeaderConfig
    topics: TeleopTopicsConfig
    bridge: TeleopBridgeConfig
    gripper: TeleopGripperConfig
    collection: TeleopCollectionConfig
    front_camera: TeleopCameraConfig
    wrist_camera: TeleopCameraConfig


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_TELEOP_CONFIG


def load_teleop_config(path: Path, *, repo_root: Path | None = None) -> TeleopConfig:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    path = path.resolve()
    data = tomllib.loads(path.read_text())
    repo_root = repo_root.resolve() if repo_root is not None else path.parents[2]

    host = _required_table(data, "host")
    leader = _required_table(data, "leader")
    topics = _required_table(data, "topics")
    bridge = _required_table(data, "bridge")
    gripper = _required_table(data, "gripper")
    collection = _required_table(data, "collection")
    cameras = _required_table(data, "cameras")
    front = _required_table(cameras, "front")
    wrist = _required_table(cameras, "wrist")

    dof = int(bridge.get("dof", 6))
    mapping = JointMappingConfig(
        relative=bool(bridge.get("relative", True)),
        input_degrees=bool(bridge.get("input_degrees", True)),
        scale=_float_tuple(bridge, "scale", dof),
        sign=_float_tuple(bridge, "sign", dof),
        bias_rad=_float_tuple(bridge, "bias_rad", dof),
        lower_limits=_float_tuple(bridge, "lower_limits", dof),
        upper_limits=_float_tuple(bridge, "upper_limits", dof),
    )
    mapping.validate(dof)

    front_width = int(front.get("width", 640))
    front_height = int(front.get("height", 480))
    front_depth_width = int(front.get("depth_width", front_width))
    front_depth_height = int(front.get("depth_height", front_height))
    wrist_width = int(wrist.get("width", 640))
    wrist_height = int(wrist.get("height", 480))

    config = TeleopConfig(
        path=path,
        host=TeleopHostConfig(
            image=_string(host, "image"),
            a1_serial=_string(host, "a1_serial"),
            prefix=_string(host, "prefix"),
            run_dir=_string(host, "run_dir"),
        ),
        leader=TeleopLeaderConfig(
            port=_string(leader, "port"),
            id=_string(leader, "id"),
            use_degrees=bool(leader.get("use_degrees", True)),
        ),
        topics=TeleopTopicsConfig(
            joint_states=_string(topics, "joint_states"),
            target=_string(topics, "target"),
            staged_command=_string(topics, "staged_command"),
            relay_enable=_string(topics, "relay_enable"),
            relay_status=_string(topics, "relay_status"),
            gripper_command=_string(topics, "gripper_command"),
            gripper_feedback=_string(topics, "gripper_feedback"),
            eef=_string(topics, "eef"),
            host_command=_string(topics, "host_command"),
        ),
        bridge=TeleopBridgeConfig(
            hz=float(bridge.get("hz", 60.0)),
            dof=dof,
            target_joint_names=_string_tuple(bridge, "target_joint_names", dof),
            mapping=mapping,
            relay_enable_timeout_s=float(bridge.get("relay_enable_timeout_s", 2.0)),
            max_relay_status_age_s=float(bridge.get("max_relay_status_age_s", 1.0)),
            a1_state_timeout_s=float(bridge.get("a1_state_timeout_s", 10.0)),
            initial_alignment_tolerance_rad=float(
                bridge.get("initial_alignment_tolerance_rad", 0.05)
            ),
        ),
        gripper=TeleopGripperConfig(
            enabled=bool(gripper.get("enabled", True)),
            source_key=_string(gripper, "source_key"),
            min_stroke_mm=float(gripper.get("min_stroke_mm", 0.0)),
            max_stroke_mm=float(gripper.get("max_stroke_mm", 200.0)),
            invert=bool(gripper.get("invert", False)),
        ),
        collection=TeleopCollectionConfig(
            data_root=_resolve_path(_string(collection, "data_root"), repo_root),
            state_mode=StateMode(_string(collection, "state_mode")),
            fps=float(collection.get("fps", 30.0)),
            max_duration_s=float(collection.get("max_duration_s", 0.0)),
            jpeg_quality=int(collection.get("jpeg_quality", 95)),
            ready_timeout_s=float(collection.get("ready_timeout_s", 10.0)),
        ),
        front_camera=TeleopCameraConfig(
            serial=str(front.get("serial", "")),
            width=front_width,
            height=front_height,
            fps=int(front.get("fps", 30)),
            depth=bool(front.get("depth", False)),
            depth_width=front_depth_width,
            depth_height=front_depth_height,
            align_depth_to_color=bool(front.get("align_depth_to_color", True)),
        ),
        wrist_camera=TeleopCameraConfig(
            device=_string(wrist, "device"),
            width=wrist_width,
            height=wrist_height,
            fps=int(wrist.get("fps", 30)),
        ),
    )
    validate_teleop_config(config)
    return config


def validate_teleop_config(config: TeleopConfig) -> None:
    if config.bridge.hz <= 0:
        raise ValueError("bridge.hz must be positive")
    if config.collection.fps <= 0:
        raise ValueError("collection.fps must be positive")
    if config.bridge.initial_alignment_tolerance_rad < 0:
        raise ValueError("bridge.initial_alignment_tolerance_rad must be non-negative")
    if config.gripper.max_stroke_mm <= config.gripper.min_stroke_mm:
        raise ValueError("gripper.max_stroke_mm must be greater than min_stroke_mm")
    for label, camera in (("front", config.front_camera), ("wrist", config.wrist_camera)):
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError(f"cameras.{label} width/height/fps must be positive")
        if camera.depth and (camera.depth_width <= 0 or camera.depth_height <= 0):
            raise ValueError(f"cameras.{label} depth_width/depth_height must be positive")
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
        "--jpeg-quality",
        str(config.collection.jpeg_quality),
        "--ready-timeout-s",
        _num(config.collection.ready_timeout_s),
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
        _bool_flag("cam0-depth-enabled", config.front_camera.depth),
        "--cam0-depth-width",
        str(config.front_camera.depth_width or config.front_camera.width),
        "--cam0-depth-height",
        str(config.front_camera.depth_height or config.front_camera.height),
        _bool_flag("cam0-align-depth-to-color", config.front_camera.align_depth_to_color),
        "--cam1-device",
        config.wrist_camera.device,
        "--cam1-width",
        str(config.wrist_camera.width),
        "--cam1-height",
        str(config.wrist_camera.height),
        "--cam1-fps",
        str(config.wrist_camera.fps),
    ]
    if config.front_camera.serial:
        args.extend(["--cam0-serial", config.front_camera.serial])
    return args


def bash_config(config: TeleopConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
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
        _array("BRIDGE_ARGS", bridge_argv(config)),
        _array("COLLECT_ARGS", collect_argv(config)),
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
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string list")
    if len(value) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(value)}")
    return tuple(value)


def _float_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    out = tuple(float(item) for item in value)
    if len(out) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(out)}")
    return out


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _csv(values: tuple[float, ...] | tuple[str, ...]) -> str:
    return ",".join(_num(value) if isinstance(value, float) else str(value) for value in values)


def _num(value: float) -> str:
    return f"{float(value):g}"


def _assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _array(name: str, values: list[str]) -> str:
    return f"{name}=({' '.join(shlex.quote(value) for value in values)})"


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
