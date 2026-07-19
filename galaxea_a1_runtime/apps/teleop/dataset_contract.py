"""Canonical direct-dataset identity derived from tracked Teleop configuration."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectDatasetIdentity,
    dataset_repo_id,
)
from galaxea_a1_runtime.schema import (
    camera_specs_from_system,
    canonical_dataset_contract,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def direct_dataset_identity(
    config: TeleopConfig, experiment: str
) -> DirectDatasetIdentity:
    """Resolve the single canonical identity used by preflight and recording."""

    return DirectDatasetIdentity(
        target_root=config.collection.dataset_root / experiment,
        repo_id=dataset_repo_id(config.collection.repo_id_prefix, experiment),
        fps=int(config.collection.fps),
        contract=canonical_dataset_contract(
            cameras=camera_specs_from_system(config.system)
        ),
        experiment=experiment,
    )


def tracked_config_reference(config: TeleopConfig, *, repo_root: Path) -> str:
    """Return a portable config identity before formal collection opens hardware."""

    try:
        return config.path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "formal collection config must be tracked inside the repository"
        ) from exc
