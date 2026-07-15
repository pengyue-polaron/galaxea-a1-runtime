"""Git-tracked teleoperation runtime configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.configuration.base import (
    float_tuple as _float_tuple,
    load_toml,
    referenced_config,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.teleop.config_runtime import (
    bash_config,
    bridge_argv,
    collect_argv,
)
from galaxea_a1_runtime.teleop.config_schema import (
    TeleopBridgeConfig,
    TeleopCollectionConfig,
    TeleopConfig,
    TeleopGripperConfig,
    TeleopLeaderConfig,
    TeleopRuntimeConfig,
)
from galaxea_a1_runtime.teleop.joint_mapping import JointMappingConfig

DEFAULT_TELEOP_CONFIG = Path("configs/teleop/a1_so100.toml")

__all__ = ["bash_config", "bridge_argv", "collect_argv", "load_teleop_config"]


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
            a1_state_timeout_s=float(
                bridge.get("a1_state_timeout_s", system.joint_safety.state_timeout_s)
            ),
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
            max_joint_action_step_rad=float(
                collection.get("max_joint_action_step_rad", 0.35)
            ),
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
        raise ValueError(
            "cameras.front.backend must be 'realsense' because teleop records optional depth framesets"
        )
    if front.depth and front.crop is not None and not front.align_depth_to_color:
        raise ValueError(
            "front depth must be aligned to color when cameras.front crop is enabled"
        )
    if len(config.system.joint_safety.names) != config.bridge.dof:
        raise ValueError("bridge.target_joint_names length must match bridge.dof")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read A1 teleop TOML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Emit bash assignments for a1_teleop_runtime.sh",
    )
    args = parser.parse_args(argv)

    config = load_teleop_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
