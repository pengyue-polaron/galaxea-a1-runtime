"""Build independently reproducible Joint and EEF LeRobot datasets from Raw v3."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.lerobot.convert_raw import convert_raw_dataset
from galaxea_a1_runtime.lerobot.eef_pack import pack_eef_v3_dataset
from galaxea_a1_runtime.lerobot.joint_pack import pack_joint_v3_dataset
from galaxea_a1_runtime.lerobot.pipeline_config import (
    DatasetPipelineConfig,
    load_pipeline_config,
)
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset

JOINT_V3 = "joint-v3"
JOINT_V21 = "joint-v2.1"
EEF_V3 = "eef-v3"
EEF_V21 = "eef-v2.1"
ALL_TARGETS = (JOINT_V3, JOINT_V21, EEF_V3, EEF_V21)


def build_datasets(
    config: DatasetPipelineConfig,
    *,
    targets: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build selected outputs from Raw v3 without reading another final output."""

    selected = _normalize_targets(targets)
    work_parent = _target_root(config, selected[0]).parent
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".a1-raw-v3-conversion-", dir=work_parent
    ) as temporary:
        workspace = Path(temporary)
        raw_lerobot_v3 = workspace / "raw-lerobot-v3"
        convert_raw_dataset(
            source_root=config.raw_source_root,
            target_root=raw_lerobot_v3,
            repo_id=_repo_id(config, selected[0]),
            overwrite=False,
            expected_contract=config.source_contract,
        )
        return {
            target: _build_target_from_raw(
                config=config,
                target=target,
                raw_lerobot_v3=raw_lerobot_v3,
                workspace=workspace,
            )
            for target in selected
        }


def _build_target_from_raw(
    *,
    config: DatasetPipelineConfig,
    target: str,
    raw_lerobot_v3: Path,
    workspace: Path,
) -> dict[str, Any]:
    raw_source = str(config.raw_source_root)
    if target == JOINT_V3:
        return pack_joint_v3_dataset(
            source_root=raw_lerobot_v3,
            target_root=config.joint_v3_target_root,
            repo_id=config.joint_v3_repo_id,
            source_dataset=raw_source,
            overwrite=config.overwrite,
            archive_path=config.joint_v3_archive_path,
        )
    if target == EEF_V3:
        return _pack_eef_v3(
            config=config,
            source_root=raw_lerobot_v3,
            target_root=config.eef_v3_target_root,
            repo_id=config.eef_v3_repo_id,
            source_dataset=raw_source,
            overwrite=config.overwrite,
            archive_path=config.eef_v3_archive_path,
        )
    if target == JOINT_V21:
        return _build_v21_from_raw(
            config=config,
            raw_lerobot_v3=raw_lerobot_v3,
            intermediate_root=workspace / "joint-v2.1-intermediate-v3",
            target_root=config.joint_v21_target_root,
            archive_path=config.joint_v21_archive_path,
            repo_id=config.joint_v21_repo_id,
            pack_v3=lambda source_root, target_root, repo_id: pack_joint_v3_dataset(
                source_root=source_root,
                target_root=target_root,
                repo_id=repo_id,
                source_dataset=raw_source,
                overwrite=False,
            ),
        )
    if target == EEF_V21:
        return _build_v21_from_raw(
            config=config,
            raw_lerobot_v3=raw_lerobot_v3,
            intermediate_root=workspace / "eef-v2.1-intermediate-v3",
            target_root=config.eef_v21_target_root,
            archive_path=config.eef_v21_archive_path,
            repo_id=config.eef_v21_repo_id,
            pack_v3=lambda source_root, target_root, repo_id: _pack_eef_v3(
                config=config,
                source_root=source_root,
                target_root=target_root,
                repo_id=repo_id,
                source_dataset=raw_source,
                overwrite=False,
                archive_path=None,
            ),
        )
    raise AssertionError(f"unhandled dataset target: {target}")


def _build_v21_from_raw(
    *,
    config: DatasetPipelineConfig,
    raw_lerobot_v3: Path,
    intermediate_root: Path,
    target_root: Path,
    archive_path: Path,
    repo_id: str,
    pack_v3: Callable[[Path, Path, str], dict[str, Any]],
) -> dict[str, Any]:
    pack_v3(raw_lerobot_v3, intermediate_root, f"{repo_id}-conversion-v3")
    return export_v21_dataset(
        source_root=intermediate_root,
        target_root=target_root,
        repo_id=repo_id,
        source_dataset=str(config.raw_source_root),
        overwrite=config.overwrite,
        archive_path=archive_path,
    )


def _pack_eef_v3(
    *,
    config: DatasetPipelineConfig,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    source_dataset: str,
    overwrite: bool,
    archive_path: Path | None,
) -> dict[str, Any]:
    return pack_eef_v3_dataset(
        source_root=source_root,
        target_root=target_root,
        urdf_path=config.urdf_path,
        repo_id=repo_id,
        gripper_stroke_min_mm=config.gripper_stroke_min_mm,
        gripper_stroke_max_mm=config.gripper_stroke_max_mm,
        base_link=config.base_link,
        tip_link=config.tip_link,
        source_dataset=source_dataset,
        overwrite=overwrite,
        archive_path=archive_path,
    )


def _normalize_targets(targets: Sequence[str] | None) -> tuple[str, ...]:
    if not targets or tuple(targets) == ("all",):
        return ALL_TARGETS
    if "all" in targets:
        raise ValueError("dataset target 'all' cannot be combined with other targets")
    unknown = sorted(set(targets) - set(ALL_TARGETS))
    if unknown:
        raise ValueError(f"unknown dataset targets: {unknown}")
    selected = set(targets)
    return tuple(target for target in ALL_TARGETS if target in selected)


def _target_root(config: DatasetPipelineConfig, target: str) -> Path:
    return {
        JOINT_V3: config.joint_v3_target_root,
        JOINT_V21: config.joint_v21_target_root,
        EEF_V3: config.eef_v3_target_root,
        EEF_V21: config.eef_v21_target_root,
    }[target]


def _repo_id(config: DatasetPipelineConfig, target: str) -> str:
    return {
        JOINT_V3: config.joint_v3_repo_id,
        JOINT_V21: config.joint_v21_repo_id,
        EEF_V3: config.eef_v3_repo_id,
        EEF_V21: config.eef_v21_repo_id,
    }[target]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        choices=("all", *ALL_TARGETS),
        help="output to build; repeat for multiple outputs (default: all)",
    )
    args = parser.parse_args(argv)
    result = build_datasets(
        load_pipeline_config(args.config),
        targets=args.target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
