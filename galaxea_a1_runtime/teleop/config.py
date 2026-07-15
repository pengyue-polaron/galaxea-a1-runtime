"""Git-tracked teleoperation runtime configuration."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.configuration.base import (
    boolean,
    float_tuple as _float_tuple,
    floating,
    integer,
    load_toml,
    referenced_config,
    require_exact_keys,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.configuration.cli import run_config_renderer
from galaxea_a1_runtime.configuration.paths import TELEOP_CONFIG
from galaxea_a1_runtime.teleop.config_runtime import bash_config
from galaxea_a1_runtime.teleop.config_schema import (
    TeleopBridgeConfig,
    TeleopCollectionConfig,
    TeleopConfig,
    TeleopGripperConfig,
    TeleopLeaderConfig,
    TeleopRuntimeConfig,
    TeleopResetConfig,
)
from galaxea_a1_runtime.teleop.joint_mapping import JointMappingConfig

DEFAULT_TELEOP_CONFIG = TELEOP_CONFIG
RUNTIME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

__all__ = ["bash_config", "load_teleop_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_TELEOP_CONFIG


def load_teleop_config(path: Path, *, repo_root: Path | None = None) -> TeleopConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "system",
            "runtime",
            "reset",
            "leader",
            "bridge",
            "gripper",
            "collection",
        },
        label="teleop config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    runtime = _required_table(data, "runtime")
    reset = _required_table(data, "reset")
    leader = _required_table(data, "leader")
    bridge = _required_table(data, "bridge")
    gripper = _required_table(data, "gripper")
    collection = _required_table(data, "collection")
    require_exact_keys(
        runtime,
        required={
            "prefix",
            "run_dir",
            "bridge_startup_timeout_s",
            "bridge_stop_timeout_s",
        },
        label="runtime",
    )
    require_exact_keys(reset, required={"config"}, label="reset")
    require_exact_keys(leader, required={"port", "id", "use_degrees"}, label="leader")
    require_exact_keys(
        bridge,
        required={
            "hz",
            "dof",
            "relative",
            "scale",
            "sign",
            "bias_rad",
            "a1_state_timeout_s",
        },
        label="bridge",
    )
    require_exact_keys(
        gripper,
        required={
            "enabled",
            "source_key",
            "source_min",
            "source_max",
            "invert",
            "saturate_out_of_range",
        },
        label="gripper",
    )
    require_exact_keys(
        collection,
        required={
            "data_root",
            "state_mode",
            "fps",
            "max_duration_s",
            "auto_reset_after_save",
            "auto_reset_after_discard",
            "jpeg_quality",
            "ready_timeout_s",
            "max_joint_action_step_rad",
        },
        label="collection",
    )
    dof = integer(bridge, "dof")
    leader_use_degrees = boolean(leader, "use_degrees")
    mapping = JointMappingConfig(
        relative=boolean(bridge, "relative"),
        input_degrees=leader_use_degrees,
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
            bridge_startup_timeout_s=floating(runtime, "bridge_startup_timeout_s"),
            bridge_stop_timeout_s=floating(runtime, "bridge_stop_timeout_s"),
        ),
        reset=TeleopResetConfig(
            config=_repo_path(repo_root, _string(reset, "config")),
        ),
        leader=TeleopLeaderConfig(
            port=_string(leader, "port"),
            id=_string(leader, "id"),
            use_degrees=leader_use_degrees,
        ),
        bridge=TeleopBridgeConfig(
            hz=floating(bridge, "hz"),
            dof=dof,
            mapping=mapping,
            a1_state_timeout_s=floating(bridge, "a1_state_timeout_s"),
        ),
        gripper=TeleopGripperConfig(
            enabled=boolean(gripper, "enabled"),
            source_key=_string(gripper, "source_key"),
            source_min=floating(gripper, "source_min"),
            source_max=floating(gripper, "source_max"),
            invert=boolean(gripper, "invert"),
            saturate_out_of_range=boolean(gripper, "saturate_out_of_range"),
        ),
        collection=TeleopCollectionConfig(
            data_root=_repo_path(repo_root, _string(collection, "data_root")),
            state_mode=StateMode(_string(collection, "state_mode")),
            fps=floating(collection, "fps"),
            max_duration_s=floating(collection, "max_duration_s"),
            auto_reset_after_save=boolean(collection, "auto_reset_after_save"),
            auto_reset_after_discard=boolean(collection, "auto_reset_after_discard"),
            jpeg_quality=integer(collection, "jpeg_quality"),
            ready_timeout_s=floating(collection, "ready_timeout_s"),
            max_joint_action_step_rad=floating(collection, "max_joint_action_step_rad"),
        ),
    )
    validate_teleop_config(config)
    return config


def validate_teleop_config(config: TeleopConfig) -> None:
    if RUNTIME_NAME.fullmatch(config.runtime.prefix) is None:
        raise ValueError("runtime.prefix must be a valid Docker resource prefix")
    if not Path(config.runtime.run_dir).is_absolute():
        raise ValueError("runtime.run_dir must be absolute")
    if config.runtime.bridge_startup_timeout_s < 1:
        raise ValueError("runtime.bridge_startup_timeout_s must be at least 1 second")
    if config.runtime.bridge_stop_timeout_s < 1:
        raise ValueError("runtime.bridge_stop_timeout_s must be at least 1 second")
    if not config.leader.port.startswith("/dev/") or any(
        character.isspace() for character in config.leader.port
    ):
        raise ValueError("leader.port must be a whitespace-free path under /dev")
    if not config.leader.use_degrees:
        raise ValueError(
            "leader.use_degrees must be true for the A1SOLeader degree mapping contract"
        )
    if config.gripper.source_key != "gripper.pos":
        raise ValueError("gripper.source_key must be 'gripper.pos' for A1SOLeader")
    if config.gripper.source_max <= config.gripper.source_min:
        raise ValueError("gripper source_max must be greater than source_min")
    if config.bridge.hz <= 0:
        raise ValueError("bridge.hz must be positive")
    if config.bridge.a1_state_timeout_s <= 0:
        raise ValueError("bridge.a1_state_timeout_s must be positive")
    if config.bridge.dof <= 0:
        raise ValueError("bridge.dof must be positive")
    if config.collection.fps <= 0:
        raise ValueError("collection.fps must be positive")
    if not config.collection.fps.is_integer():
        raise ValueError("collection.fps must be an integer for LeRobot conversion")
    if config.collection.max_duration_s < 0:
        raise ValueError("collection.max_duration_s must be non-negative")
    if not 1 <= config.collection.jpeg_quality <= 100:
        raise ValueError("collection.jpeg_quality must be in [1, 100]")
    if config.collection.ready_timeout_s <= 0:
        raise ValueError("collection.ready_timeout_s must be positive")
    if config.collection.max_joint_action_step_rad <= 0:
        raise ValueError("collection.max_joint_action_step_rad must be positive")
    front = config.system.cameras.front
    if front.backend != "realsense":
        raise ValueError(
            "cameras.front.backend must be 'realsense' because teleop records optional depth framesets"
        )
    if len(config.system.joint_safety.names) != config.bridge.dof:
        raise ValueError("bridge.target_joint_names length must match bridge.dof")


def main(argv: list[str] | None = None) -> int:
    return run_config_renderer(
        argv,
        description="Read the tracked A1 teleoperation config.",
        default_config=DEFAULT_TELEOP_CONFIG,
        load_config=load_teleop_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
