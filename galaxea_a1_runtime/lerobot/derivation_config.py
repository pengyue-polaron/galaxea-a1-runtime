"""Tracked configuration for derivatives of a canonical direct LeRobot dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    boolean,
    load_toml,
    paths_overlap,
    referenced_config,
    repo_path,
    require_exact_keys,
    required_table,
    string,
)
from galaxea_a1_runtime.configuration.system import SystemConfig, load_system_config
from galaxea_a1_runtime.lerobot.dataset import validate_dataset_repo_id


@dataclass(frozen=True)
class DerivativeOutput:
    target_root: Path
    archive_path: Path
    repo_id: str


@dataclass(frozen=True)
class DirectDerivationConfig:
    system: SystemConfig
    source_root: Path
    overwrite: bool
    joint_v21: DerivativeOutput
    eef_v3: DerivativeOutput
    eef_v21: DerivativeOutput
    urdf_path: Path
    base_link: str
    tip_link: str


def load_derivation_config(
    path: Path, *, repo_root: Path | None = None
) -> DirectDerivationConfig:
    _, repo_root, raw = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        raw,
        required={"system", "derivation", "source", "outputs", "kinematics"},
        label="direct dataset derivation config",
    )
    system_reference = required_table(raw, "system")
    derivation = required_table(raw, "derivation")
    source = required_table(raw, "source")
    outputs = required_table(raw, "outputs")
    kinematics = required_table(raw, "kinematics")
    require_exact_keys(system_reference, required={"config"}, label="system")
    require_exact_keys(derivation, required={"overwrite"}, label="derivation")
    require_exact_keys(source, required={"root"}, label="source")
    require_exact_keys(
        outputs,
        required={"joint_v21", "eef_v3", "eef_v21"},
        label="outputs",
    )
    require_exact_keys(
        kinematics,
        required={"urdf", "base_link", "tip_link"},
        label="kinematics",
    )
    config = DirectDerivationConfig(
        system=load_system_config(
            referenced_config(raw, repo_root), repo_root=repo_root
        ),
        source_root=repo_path(repo_root, string(source, "root")),
        overwrite=boolean(derivation, "overwrite"),
        joint_v21=_output(outputs, "joint_v21", repo_root=repo_root),
        eef_v3=_output(outputs, "eef_v3", repo_root=repo_root),
        eef_v21=_output(outputs, "eef_v21", repo_root=repo_root),
        urdf_path=repo_path(repo_root, string(kinematics, "urdf")),
        base_link=string(kinematics, "base_link"),
        tip_link=string(kinematics, "tip_link"),
    )
    _validate_paths(config)
    return config


def _output(
    outputs: dict,
    name: str,
    *,
    repo_root: Path,
) -> DerivativeOutput:
    value = required_table(outputs, name)
    require_exact_keys(
        value,
        required={"target_root", "archive_path", "repo_id"},
        label=f"outputs.{name}",
    )
    repo_id = string(value, "repo_id")
    validate_dataset_repo_id(repo_id, label=f"outputs.{name}.repo_id")
    return DerivativeOutput(
        target_root=repo_path(repo_root, string(value, "target_root")),
        archive_path=repo_path(repo_root, string(value, "archive_path")),
        repo_id=repo_id,
    )


def _validate_paths(config: DirectDerivationConfig) -> None:
    targets = (
        config.joint_v21.target_root,
        config.eef_v3.target_root,
        config.eef_v21.target_root,
    )
    archives = (
        config.joint_v21.archive_path,
        config.eef_v3.archive_path,
        config.eef_v21.archive_path,
    )
    if len(set(targets)) != len(targets):
        raise ValueError("direct derivative target roots must be unique")
    if any(
        paths_overlap(left, right)
        for index, left in enumerate(targets)
        for right in targets[index + 1 :]
    ):
        raise ValueError("direct derivative target roots must not overlap")
    if len(set(archives)) != len(archives):
        raise ValueError("direct derivative archive paths must be unique")
    if any(paths_overlap(config.source_root, target) for target in targets):
        raise ValueError("canonical source must not overlap a derivative root")
    if any(
        archive.is_relative_to(target) for archive in archives for target in targets
    ):
        raise ValueError("derivative archives must be outside derivative roots")
