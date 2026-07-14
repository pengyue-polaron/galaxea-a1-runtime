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
from galaxea_a1_runtime.hardware.web_preview import web_preview_argv
from galaxea_a1_runtime.teleop.joint_mapping import JointMappingConfig

DEFAULT_TELEOP_CONFIG = Path("configs/teleop/a1_so100.toml")


@dataclass(frozen=True)
class TeleopRuntimeConfig:
    prefix: str
    run_dir: str


@dataclass(frozen=True)
class TeleopLeaderConfig:
    port: str
    id: str
    use_degrees: bool


@dataclass(frozen=True)
class TeleopBridgeConfig:
    hz: float
    dof: int
    mapping: JointMappingConfig
    a1_state_timeout_s: float


@dataclass(frozen=True)
class TeleopGripperConfig:
    enabled: bool
    source_key: str
    invert: bool


@dataclass(frozen=True)
class TeleopCollectionConfig:
    data_root: Path
    state_mode: StateMode
    fps: float
    max_duration_s: float
    auto_reset_after_save: bool
    jpeg_quality: int
    ready_timeout_s: float
    max_joint_action_step_rad: float


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    system: SystemConfig
    runtime: TeleopRuntimeConfig
    leader: TeleopLeaderConfig
    bridge: TeleopBridgeConfig
    gripper: TeleopGripperConfig
    collection: TeleopCollectionConfig


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
        runtime=TeleopRuntimeConfig(
            prefix=_string(runtime, "prefix"),
            run_dir=_string(runtime, "run_dir"),
        ),
        leader=TeleopLeaderConfig(
            port=_string(leader, "port"),
            id=_string(leader, "id"),
            use_degrees=bool(leader.get("use_degrees", True)),
        ),
        bridge=TeleopBridgeConfig(
            hz=float(bridge.get("hz", 60.0)),
            dof=dof,
            mapping=mapping,
            a1_state_timeout_s=float(bridge.get("a1_state_timeout_s", system.joint_safety.state_timeout_s)),
        ),
        gripper=TeleopGripperConfig(
            enabled=bool(gripper.get("enabled", True)),
            source_key=_string(gripper, "source_key"),
            invert=bool(gripper.get("invert", False)),
        ),
        collection=TeleopCollectionConfig(
            data_root=_repo_path(repo_root, _string(collection, "data_root")),
            state_mode=StateMode(_string(collection, "state_mode")),
            fps=float(collection.get("fps", 30.0)),
            max_duration_s=float(collection.get("max_duration_s", 0.0)),
            auto_reset_after_save=bool(collection.get("auto_reset_after_save", True)),
            jpeg_quality=int(collection.get("jpeg_quality", 95)),
            ready_timeout_s=float(collection.get("ready_timeout_s", 10.0)),
            max_joint_action_step_rad=float(collection.get("max_joint_action_step_rad", 0.35)),
        ),
    )
    validate_teleop_config(config)
    return config


def validate_teleop_config(config: TeleopConfig) -> None:
    if config.bridge.hz <= 0:
        raise ValueError("bridge.hz must be positive")
    if config.collection.fps <= 0:
        raise ValueError("collection.fps must be positive")
    if config.collection.max_joint_action_step_rad <= 0:
        raise ValueError("collection.max_joint_action_step_rad must be positive")
    front = config.system.cameras.front
    if front.backend != "realsense":
        raise ValueError("cameras.front.backend must be 'realsense' because teleop records optional depth framesets")
    if front.depth and front.crop is not None and not front.align_depth_to_color:
        raise ValueError("front depth must be aligned to color when cameras.front crop is enabled")
    if len(config.system.joint_safety.names) != config.bridge.dof:
        raise ValueError("bridge.target_joint_names length must match bridge.dof")


def bridge_argv(config: TeleopConfig) -> list[str]:
    mapping = config.bridge.mapping
    system = config.system
    topics = system.topics
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
        topics.joint_states,
        "--target-topic",
        topics.joint_target,
        "--staged-command-topic",
        topics.staged_command,
        "--target-joint-names",
        _csv(system.joint_safety.names),
        _bool_flag("relative", mapping.relative),
        _bool_flag("input-degrees", mapping.input_degrees),
        f"--scale={_csv(mapping.scale)}",
        f"--sign={_csv(mapping.sign)}",
        f"--bias-rad={_csv(mapping.bias_rad)}",
        f"--lower-limits={_csv(mapping.lower_limits)}",
        f"--upper-limits={_csv(mapping.upper_limits)}",
        "--motion-enable-topic",
        topics.motion_enable,
        "--relay-status-topic",
        topics.relay_status,
        "--relay-enable-timeout",
        _num(system.relay.enable_timeout_s),
        "--max-relay-status-age",
        _num(system.relay.max_status_age_s),
        "--a1-state-timeout",
        _num(config.bridge.a1_state_timeout_s),
        "--initial-alignment-tolerance",
        _num(system.joint_safety.initial_alignment_tolerance_rad),
        _bool_flag("gripper-enabled", config.gripper.enabled),
        "--gripper-source-key",
        config.gripper.source_key,
        "--gripper-topic",
        topics.gripper_target,
        "--gripper-min-stroke-mm",
        _num(system.gripper.stroke_min_mm),
        "--gripper-max-stroke-mm",
        _num(system.gripper.stroke_max_mm),
    ]
    if config.gripper.invert:
        args.append("--gripper-invert")
    return args


def collect_argv(config: TeleopConfig) -> list[str]:
    system = config.system
    topics = system.topics
    front = system.cameras.front
    wrist = system.cameras.wrist
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
        _num(system.cameras.max_age_s),
        "--max-joint-feedback-age-s",
        _num(system.joint_safety.max_feedback_age_s),
        "--max-eef-feedback-age-s",
        _num(system.eef.max_feedback_age_s),
        "--max-action-age-s",
        _num(system.joint_safety.max_feedback_age_s),
        "--max-gripper-age-s",
        _num(system.joint_safety.max_feedback_age_s),
        "--max-joint-action-step-rad",
        _num(config.collection.max_joint_action_step_rad),
        "--joint-topic",
        topics.joint_states,
        "--eef-topic",
        topics.eef_pose,
        "--action-topic",
        topics.joint_target,
        "--gripper-feedback-topic",
        topics.gripper_feedback,
        "--gripper-action-topic",
        topics.gripper_target,
        "--gripper-stroke-min",
        _num(system.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        _num(system.gripper.stroke_max_mm),
        "--staged-command-topic",
        topics.staged_command,
        "--host-command-topic",
        topics.host_command,
        "--cam0-width",
        str(front.width),
        "--cam0-height",
        str(front.height),
        "--cam0-fps",
        str(front.fps),
        _bool_flag("cam0-require-usb3", front.require_usb3),
        _bool_flag("cam0-depth-enabled", front.depth),
        "--cam0-depth-width",
        str(front.depth_width or front.width),
        "--cam0-depth-height",
        str(front.depth_height or front.height),
        _bool_flag("cam0-align-depth-to-color", front.align_depth_to_color),
        _bool_flag("cam0-crop-enabled", front.crop is not None),
        "--cam1-device",
        wrist.device,
        "--cam1-backend",
        wrist.backend,
        "--cam1-serial",
        wrist.serial,
        "--cam1-width",
        str(wrist.width),
        "--cam1-height",
        str(wrist.height),
        "--cam1-fps",
        str(wrist.fps),
        "--cam1-pixel-format",
        wrist.pixel_format,
        *web_preview_argv(system.web_preview),
    ]
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
    args.extend(["--cam0-serial", front.serial])
    return args


def bash_config(config: TeleopConfig) -> str:
    lines = [
        _assign("CONFIG_PATH", str(config.path)),
        _assign("SYSTEM_CONFIG_PATH", str(config.system.path)),
        _assign("IMAGE", config.system.host.image),
        _assign("SERIAL", config.system.host.a1_serial),
        _assign("LEADER_PORT", config.leader.port),
        _assign("LEADER_ID", config.leader.id),
        _assign("PREFIX", config.runtime.prefix),
        _assign("RUN_DIR", config.runtime.run_dir),
        _assign("STAGED_TOPIC", config.system.topics.staged_command),
        _assign("RELAY_ENABLE_TOPIC", config.system.topics.motion_enable),
        _assign("RELAY_STATUS_TOPIC", config.system.topics.relay_status),
        _assign("JOINT_STATES_TOPIC", config.system.topics.joint_states),
        _assign("HOST_COMMAND_TOPIC", config.system.topics.host_command),
        _assign("MOTOR_STATUS_TOPIC", config.system.topics.motor_status),
        _assign("GRIPPER_TARGET_TOPIC", config.system.topics.gripper_target),
        _assign("GRIPPER_COMMAND_TOPIC", config.system.topics.gripper_command),
        _assign("EEF_POSE_TOPIC", config.system.topics.eef_pose),
        _assign("RELAY_MAX_INPUT_AGE_S", _num(config.system.relay.max_input_age_s)),
        _assign("RELAY_ARMING_TIMEOUT_S", _num(config.system.relay.arming_timeout_s)),
        _assign(
            "RELAY_MAX_INITIAL_ERROR_RAD",
            _num(config.system.joint_safety.initial_alignment_tolerance_rad),
        ),
        _assign("TARGET_TOPIC", config.system.topics.joint_target),
        _assign("GRIPPER_MIN_STROKE_MM", _num(config.system.gripper.stroke_min_mm)),
        _assign("GRIPPER_MAX_STROKE_MM", _num(config.system.gripper.stroke_max_mm)),
        _assign("WEB_PREVIEW_PORT", str(config.system.web_preview.port)),
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
