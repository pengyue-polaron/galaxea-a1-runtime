"""Tracked dataset packaging contract for A1 LeRobot and policy exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    load_toml,
    repo_path,
    require_exact_keys,
    required_table,
    string,
)
from galaxea_a1_runtime.collection import state_names_for_mode
from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import (
    ActionMode,
    JOINT_ACTION_NAMES,
    DatasetContract,
    camera_specs_from_system,
)
from galaxea_a1_runtime.teleop.config import load_teleop_config


@dataclass(frozen=True)
class LingBotPackConfig:
    raw_source_root: Path
    base_v3_root: Path
    base_v3_repo_id: str
    base_v21_target_root: Path
    base_v21_archive_path: Path
    base_v21_repo_id: str
    overwrite: bool
    source_contract: DatasetContract
    eef_v3_target_root: Path
    eef_v3_archive_path: Path
    eef_v3_repo_id: str
    eef_v21_target_root: Path
    eef_v21_archive_path: Path
    eef_v21_repo_id: str
    joint_v3_target_root: Path
    joint_v3_archive_path: Path
    joint_v3_repo_id: str
    urdf_path: Path
    base_link: str
    tip_link: str
    gripper_stroke_min_mm: float
    gripper_stroke_max_mm: float


def load_pack_config(path: Path) -> LingBotPackConfig:
    _, repo_root, raw = load_toml(path)
    require_exact_keys(
        raw,
        required={"teleop", "dataset", "outputs", "kinematics"},
        label="dataset pack config",
    )
    teleop_reference = required_table(raw, "teleop")
    require_exact_keys(
        teleop_reference,
        required={"config"},
        label="teleop reference",
    )
    teleop = load_teleop_config(
        repo_path(repo_root, string(teleop_reference, "config")),
        repo_root=repo_root,
    )
    system = teleop.system
    dataset = required_table(raw, "dataset")
    outputs = required_table(raw, "outputs")
    kinematics = required_table(raw, "kinematics")
    require_exact_keys(
        dataset,
        required={
            "raw_root",
            "lerobot_v3_root",
            "repo_id",
            "overwrite",
        },
        label="dataset",
    )
    require_exact_keys(
        outputs,
        required={"base_v21", "eef_v3", "eef_v21", "joint_v3"},
        label="outputs",
    )
    require_exact_keys(
        kinematics,
        required={"urdf", "base_link", "tip_link"},
        label="kinematics",
    )
    base_v21 = _output_config(outputs, "base_v21")
    eef_v3 = _output_config(outputs, "eef_v3")
    eef_v21 = _output_config(outputs, "eef_v21")
    joint_v3 = _output_config(outputs, "joint_v3")
    config = LingBotPackConfig(
        raw_source_root=repo_path(repo_root, string(dataset, "raw_root")),
        base_v3_root=repo_path(repo_root, string(dataset, "lerobot_v3_root")),
        base_v3_repo_id=string(dataset, "repo_id"),
        base_v21_target_root=repo_path(repo_root, string(base_v21, "target_root")),
        base_v21_archive_path=repo_path(repo_root, string(base_v21, "archive_path")),
        base_v21_repo_id=string(base_v21, "repo_id"),
        overwrite=boolean(dataset, "overwrite"),
        source_contract=DatasetContract(
            dataset_format=LEROBOT_DATASET_FORMAT,
            action_mode=ActionMode.JOINT_ABSOLUTE,
            state_names=state_names_for_mode(teleop.collection.state_mode),
            action_names=JOINT_ACTION_NAMES,
            camera_specs=camera_specs_from_system(system),
        ),
        eef_v3_target_root=repo_path(repo_root, string(eef_v3, "target_root")),
        eef_v3_archive_path=repo_path(repo_root, string(eef_v3, "archive_path")),
        eef_v3_repo_id=string(eef_v3, "repo_id"),
        eef_v21_target_root=repo_path(repo_root, string(eef_v21, "target_root")),
        eef_v21_archive_path=repo_path(repo_root, string(eef_v21, "archive_path")),
        eef_v21_repo_id=string(eef_v21, "repo_id"),
        joint_v3_target_root=repo_path(repo_root, string(joint_v3, "target_root")),
        joint_v3_archive_path=repo_path(repo_root, string(joint_v3, "archive_path")),
        joint_v3_repo_id=string(joint_v3, "repo_id"),
        urdf_path=repo_path(repo_root, string(kinematics, "urdf")),
        base_link=string(kinematics, "base_link"),
        tip_link=string(kinematics, "tip_link"),
        gripper_stroke_min_mm=system.gripper.stroke_min_mm,
        gripper_stroke_max_mm=system.gripper.stroke_max_mm,
    )
    _validate_pack_paths(config)
    return config


def _output_config(outputs: dict[str, Any], name: str) -> dict[str, Any]:
    value = required_table(outputs, name)
    require_exact_keys(
        value,
        required={"target_root", "archive_path", "repo_id"},
        label=f"outputs.{name}",
    )
    return value


def _validate_pack_paths(config: LingBotPackConfig) -> None:
    targets = (
        config.base_v21_target_root,
        config.eef_v3_target_root,
        config.eef_v21_target_root,
        config.joint_v3_target_root,
    )
    archives = (
        config.base_v21_archive_path,
        config.eef_v3_archive_path,
        config.eef_v21_archive_path,
        config.joint_v3_archive_path,
    )
    if len(set(targets)) != len(targets):
        raise ValueError("dataset output target roots must be unique")
    if len(set(archives)) != len(archives):
        raise ValueError("dataset output archive paths must be unique")
    if config.base_v3_root in targets:
        raise ValueError("base v3 root must differ from every output target")
    if config.raw_source_root in (*targets, config.base_v3_root):
        raise ValueError("raw source must differ from every processed dataset root")
