"""Git-tracked teleoperation runtime configuration."""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
    JointMappingConfig,
    TeleopBridgeConfig,
    TeleopCollectionConfig,
    TeleopConfig,
    TeleopGripperConfig,
    TeleopLeaderConfig,
    TeleopRuntimeConfig,
    TeleopResetConfig,
)

RUNTIME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

__all__ = ["bash_config", "load_teleop_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / TELEOP_CONFIG


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
    require_exact_keys(
        leader,
        required={"port", "id", "motor_write_retries"},
        label="leader",
    )
    require_exact_keys(
        bridge,
        required={
            "hz",
            "scale",
            "sign",
            "bias_rad",
        },
        label="bridge",
    )
    require_exact_keys(
        gripper,
        required={
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
            "dataset_root",
            "repo_id_prefix",
            "fps",
            "max_duration_s",
            "auto_reset_after_save",
            "auto_reset_after_discard",
            "ready_timeout_s",
            "max_joint_action_step_rad",
        },
        label="collection",
    )
    dof = len(system.joint_safety.names)
    mapping = JointMappingConfig(
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
            motor_write_retries=integer(leader, "motor_write_retries"),
        ),
        bridge=TeleopBridgeConfig(
            hz=floating(bridge, "hz"),
            mapping=mapping,
        ),
        gripper=TeleopGripperConfig(
            source_min=floating(gripper, "source_min"),
            source_max=floating(gripper, "source_max"),
            invert=boolean(gripper, "invert"),
            saturate_out_of_range=boolean(gripper, "saturate_out_of_range"),
        ),
        collection=TeleopCollectionConfig(
            dataset_root=_repo_path(repo_root, _string(collection, "dataset_root")),
            repo_id_prefix=_string(collection, "repo_id_prefix"),
            fps=floating(collection, "fps"),
            max_duration_s=floating(collection, "max_duration_s"),
            auto_reset_after_save=boolean(collection, "auto_reset_after_save"),
            auto_reset_after_discard=boolean(collection, "auto_reset_after_discard"),
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
    if config.leader.motor_write_retries < 1:
        raise ValueError("leader.motor_write_retries must be at least 1")
    if config.gripper.source_max <= config.gripper.source_min:
        raise ValueError("gripper source_max must be greater than source_min")
    if config.bridge.hz <= 0:
        raise ValueError("bridge.hz must be positive")
    minimum_startup_timeout_s = (
        2 * config.system.embodied_ops.device_connect_timeout_s
        + config.system.relay.enable_timeout_s
    )
    if config.runtime.bridge_startup_timeout_s < minimum_startup_timeout_s:
        raise ValueError(
            "runtime.bridge_startup_timeout_s must cover two System "
            "embodied_ops.device_connect_timeout_s windows plus the relay enable timeout"
        )
    if config.collection.fps <= 0:
        raise ValueError("collection.fps must be positive")
    if not config.collection.fps.is_integer():
        raise ValueError("collection.fps must be an integer for LeRobot recording")
    if config.collection.max_duration_s < 0:
        raise ValueError("collection.max_duration_s must be non-negative")
    prefix = config.collection.repo_id_prefix
    if prefix.count("/") != 1 or any(character.isspace() for character in prefix):
        raise ValueError(
            "collection.repo_id_prefix must be a whitespace-free 'owner/name' prefix"
        )
    if config.collection.ready_timeout_s <= 0:
        raise ValueError("collection.ready_timeout_s must be positive")
    if config.collection.max_joint_action_step_rad <= 0:
        raise ValueError("collection.max_joint_action_step_rad must be positive")


def validate_collection_config(config: TeleopConfig) -> None:
    """Validate collection-only hardware contracts before any device startup."""

    if config.system.cameras.front.backend != "realsense":
        raise ValueError(
            "cameras.front.backend must be 'realsense' for canonical Teleop collection"
        )


def main(argv: list[str] | None = None) -> int:
    return run_config_renderer(
        argv,
        description="Read the tracked A1 teleoperation config.",
        default_config=TELEOP_CONFIG,
        load_config=load_teleop_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
