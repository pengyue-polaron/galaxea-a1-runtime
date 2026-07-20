"""Tracked contract for the generic A1 LeRobot conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    load_toml,
    paths_overlap,
    repo_path,
    require_exact_keys,
    required_table,
    string,
)
from galaxea_a1_runtime.collection.schema import validate_experiment_name
from galaxea_a1_runtime.lerobot.boundary_trim_config import BoundaryTrimConfig
from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import (
    ActionMode,
    LEGACY_RAW_ACTION_NAMES,
    LEGACY_RAW_STATE_NAMES,
    DatasetContract,
    camera_specs_from_system,
)
from galaxea_a1_runtime.teleop.config import load_teleop_config


@dataclass(frozen=True)
class DatasetPipelineConfig:
    raw_source_id: str
    raw_source_roots: tuple[Path, ...]
    overwrite: bool
    boundary_trim: BoundaryTrimConfig
    source_contract: DatasetContract
    joint_v3_target_root: Path
    joint_v3_archive_path: Path
    joint_v3_repo_id: str
    joint_v21_target_root: Path
    joint_v21_archive_path: Path
    joint_v21_repo_id: str
    eef_v3_target_root: Path
    eef_v3_archive_path: Path
    eef_v3_repo_id: str
    eef_v21_target_root: Path
    eef_v21_archive_path: Path
    eef_v21_repo_id: str
    urdf_path: Path
    base_link: str
    tip_link: str
    gripper_stroke_min_mm: float
    gripper_stroke_max_mm: float


def load_pipeline_config(path: Path) -> DatasetPipelineConfig:
    _, repo_root, raw = load_toml(path)
    require_exact_keys(
        raw,
        required={"teleop", "dataset", "trim", "outputs", "kinematics"},
        label="dataset pipeline config",
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
    trim = required_table(raw, "trim")
    outputs = required_table(raw, "outputs")
    kinematics = required_table(raw, "kinematics")
    require_exact_keys(
        dataset,
        required={"raw_source_id", "raw_source_roots", "overwrite"},
        label="dataset",
    )
    raw_source_id = string(dataset, "raw_source_id")
    if raw_source_id.count("/") != 1 or any(
        not part for part in raw_source_id.split("/")
    ):
        raise ValueError("dataset.raw_source_id must be a namespaced dataset ID")
    raw_source_roots = _raw_source_roots(dataset, repo_root=repo_root)
    require_exact_keys(
        trim,
        required={
            "enabled",
            "anchor_window_s",
            "joint_deadband_rad",
            "gripper_deadband",
            "confirm_frames",
            "pre_roll_s",
            "post_roll_s",
            "max_trim_fraction",
            "min_kept_duration_s",
        },
        label="trim",
    )
    require_exact_keys(
        outputs,
        required={"joint_v3", "joint_v21", "eef_v3", "eef_v21"},
        label="outputs",
    )
    require_exact_keys(
        kinematics,
        required={"urdf", "base_link", "tip_link"},
        label="kinematics",
    )
    joint_v3 = _output_config(outputs, "joint_v3")
    joint_v21 = _output_config(outputs, "joint_v21")
    eef_v3 = _output_config(outputs, "eef_v3")
    eef_v21 = _output_config(outputs, "eef_v21")
    config = DatasetPipelineConfig(
        raw_source_id=raw_source_id,
        raw_source_roots=raw_source_roots,
        overwrite=boolean(dataset, "overwrite"),
        boundary_trim=_boundary_trim_config(trim),
        source_contract=DatasetContract(
            dataset_format=LEROBOT_DATASET_FORMAT,
            action_mode=ActionMode.JOINT_ABSOLUTE,
            state_names=LEGACY_RAW_STATE_NAMES,
            action_names=LEGACY_RAW_ACTION_NAMES,
            camera_specs=camera_specs_from_system(system),
        ),
        joint_v3_target_root=repo_path(repo_root, string(joint_v3, "target_root")),
        joint_v3_archive_path=repo_path(repo_root, string(joint_v3, "archive_path")),
        joint_v3_repo_id=string(joint_v3, "repo_id"),
        joint_v21_target_root=repo_path(repo_root, string(joint_v21, "target_root")),
        joint_v21_archive_path=repo_path(repo_root, string(joint_v21, "archive_path")),
        joint_v21_repo_id=string(joint_v21, "repo_id"),
        eef_v3_target_root=repo_path(repo_root, string(eef_v3, "target_root")),
        eef_v3_archive_path=repo_path(repo_root, string(eef_v3, "archive_path")),
        eef_v3_repo_id=string(eef_v3, "repo_id"),
        eef_v21_target_root=repo_path(repo_root, string(eef_v21, "target_root")),
        eef_v21_archive_path=repo_path(repo_root, string(eef_v21, "archive_path")),
        eef_v21_repo_id=string(eef_v21, "repo_id"),
        urdf_path=repo_path(repo_root, string(kinematics, "urdf")),
        base_link=string(kinematics, "base_link"),
        tip_link=string(kinematics, "tip_link"),
        gripper_stroke_min_mm=system.gripper.stroke_min_mm,
        gripper_stroke_max_mm=system.gripper.stroke_max_mm,
    )
    _validate_pack_paths(config)
    return config


def _boundary_trim_config(data: dict[str, Any]) -> BoundaryTrimConfig:
    return BoundaryTrimConfig(
        enabled=boolean(data, "enabled"),
        anchor_window_s=floating(data, "anchor_window_s"),
        joint_deadband_rad=floating(data, "joint_deadband_rad"),
        gripper_deadband=floating(data, "gripper_deadband"),
        confirm_frames=integer(data, "confirm_frames"),
        pre_roll_s=floating(data, "pre_roll_s"),
        post_roll_s=floating(data, "post_roll_s"),
        max_trim_fraction=floating(data, "max_trim_fraction"),
        min_kept_duration_s=floating(data, "min_kept_duration_s"),
    )


def _raw_source_roots(data: dict[str, Any], *, repo_root: Path) -> tuple[Path, ...]:
    values = data.get("raw_source_roots")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise ValueError("dataset.raw_source_roots must be a non-empty string list")
    roots = tuple(repo_path(repo_root, value) for value in values)
    if len(set(roots)) != len(roots):
        raise ValueError("dataset.raw_source_roots contains duplicates")
    raw_root = (repo_root / "data" / "raw").resolve()
    for root in roots:
        if not root.is_relative_to(raw_root):
            raise ValueError(f"legacy Raw v3 source must be below {raw_root}: {root}")
        validate_experiment_name(root.name)
    return roots


def _output_config(outputs: dict[str, Any], name: str) -> dict[str, Any]:
    value = required_table(outputs, name)
    require_exact_keys(
        value,
        required={"target_root", "archive_path", "repo_id"},
        label=f"outputs.{name}",
    )
    return value


def _validate_pack_paths(config: DatasetPipelineConfig) -> None:
    targets = (
        config.joint_v3_target_root,
        config.joint_v21_target_root,
        config.eef_v3_target_root,
        config.eef_v21_target_root,
    )
    archives = (
        config.joint_v3_archive_path,
        config.joint_v21_archive_path,
        config.eef_v3_archive_path,
        config.eef_v21_archive_path,
    )
    if len(set(targets)) != len(targets):
        raise ValueError("dataset output target roots must be unique")
    if any(
        paths_overlap(left, right)
        for index, left in enumerate(targets)
        for right in targets[index + 1 :]
    ):
        raise ValueError("dataset output target roots must not overlap")
    if len(set(archives)) != len(archives):
        raise ValueError("dataset output archive paths must be unique")
    if any(
        paths_overlap(source_root, target)
        for source_root in config.raw_source_roots
        for target in targets
    ):
        raise ValueError("raw source must not overlap a processed dataset root")
    if any(
        archive.is_relative_to(target) for archive in archives for target in targets
    ):
        raise ValueError("dataset archives must be outside processed dataset roots")
