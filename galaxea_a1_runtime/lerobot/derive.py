"""Build deterministic derivatives from a canonical direct LeRobotDataset v3."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.lerobot.derivation_config import (
    DerivativeOutput,
    DirectDerivationConfig,
    load_derivation_config,
)
from galaxea_a1_runtime.lerobot.direct_recording import discover_direct_dataset
from galaxea_a1_runtime.lerobot.eef_pack import pack_eef_v3_dataset
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset
from galaxea_a1_runtime.schema import (
    camera_specs_from_system,
    canonical_dataset_contract,
)

JOINT_V21 = "joint-v2.1"
EEF_V3 = "eef-v3"
EEF_V21 = "eef-v2.1"
ALL_TARGETS = (JOINT_V21, EEF_V3, EEF_V21)


def build_derivatives(
    config: DirectDerivationConfig,
    *,
    targets: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build each selected output directly from the canonical source."""

    selected = _normalize_targets(targets)
    source = discover_direct_dataset(
        config.source_root,
        contract=canonical_dataset_contract(
            cameras=camera_specs_from_system(config.system)
        ),
    )
    source_id = source.identity.repo_id
    return {
        target: _build_target(
            config=config,
            target=target,
            source_root=source.identity.target_root,
            source_id=source_id,
        )
        for target in selected
    }


def _build_target(
    *,
    config: DirectDerivationConfig,
    target: str,
    source_root: Path,
    source_id: str,
) -> dict[str, Any]:
    output = _output(config, target)
    if target == EEF_V3:
        return _pack_eef(
            config=config,
            source_root=source_root,
            target_root=output.target_root,
            repo_id=output.repo_id,
            source_id=source_id,
            overwrite=config.overwrite,
            archive_path=output.archive_path,
        )
    if target == JOINT_V21:
        return export_v21_dataset(
            source_root=source_root,
            target_root=output.target_root,
            repo_id=output.repo_id,
            source_dataset=source_id,
            overwrite=config.overwrite,
            archive_path=output.archive_path,
        )
    if target == EEF_V21:
        work_parent = output.target_root.parent
        work_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".a1-eef-v2.1-derivation-", dir=work_parent
        ) as temporary:
            intermediate = Path(temporary) / "eef-v3"
            _pack_eef(
                config=config,
                source_root=source_root,
                target_root=intermediate,
                repo_id=f"{output.repo_id}-conversion-v3",
                source_id=source_id,
                overwrite=False,
                archive_path=None,
            )
            return export_v21_dataset(
                source_root=intermediate,
                target_root=output.target_root,
                repo_id=output.repo_id,
                source_dataset=source_id,
                overwrite=config.overwrite,
                archive_path=output.archive_path,
            )
    raise AssertionError(f"unhandled derivative target: {target}")


def _pack_eef(
    *,
    config: DirectDerivationConfig,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    source_id: str,
    overwrite: bool,
    archive_path: Path | None,
) -> dict[str, Any]:
    return pack_eef_v3_dataset(
        source_root=source_root,
        target_root=target_root,
        urdf_path=config.urdf_path,
        repo_id=repo_id,
        gripper_stroke_min_mm=config.system.gripper.stroke_min_mm,
        gripper_stroke_max_mm=config.system.gripper.stroke_max_mm,
        base_link=config.base_link,
        tip_link=config.tip_link,
        source_dataset=source_id,
        overwrite=overwrite,
        archive_path=archive_path,
    )


def _normalize_targets(targets: Sequence[str] | None) -> tuple[str, ...]:
    if not targets or tuple(targets) == ("all",):
        return ALL_TARGETS
    if "all" in targets:
        raise ValueError(
            "derivative target 'all' cannot be combined with other targets"
        )
    unknown = sorted(set(targets) - set(ALL_TARGETS))
    if unknown:
        raise ValueError(f"unknown derivative targets: {unknown}")
    selected = set(targets)
    return tuple(target for target in ALL_TARGETS if target in selected)


def _output(config: DirectDerivationConfig, target: str) -> DerivativeOutput:
    return {
        JOINT_V21: config.joint_v21,
        EEF_V3: config.eef_v3,
        EEF_V21: config.eef_v21,
    }[target]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        choices=("all", *ALL_TARGETS),
        help="derivative to build; repeat for multiple outputs (default: all)",
    )
    args = parser.parse_args(argv)
    result = build_derivatives(
        load_derivation_config(args.config),
        targets=args.target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
