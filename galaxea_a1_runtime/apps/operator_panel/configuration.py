"""A1 configuration kinds and their owning strict validators."""

from __future__ import annotations

import tomllib
from functools import partial
from pathlib import Path

from operator_panel.config_store import ConfigKind, RepositoryConfigStore

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.teleop.config import load_teleop_config


def build_a1_config_store(repo_root: Path) -> RepositoryConfigStore:
    root = repo_root.resolve()
    return RepositoryConfigStore(
        root,
        (
            ConfigKind(
                "teleop",
                "Teleop",
                Path("configs/teleop"),
                partial(_validate_teleop, root),
            ),
            ConfigKind(
                "deployment",
                "LingBot deployment",
                Path("configs/deployments/lingbot"),
                partial(_validate_deployment, root),
            ),
            ConfigKind(
                "batch",
                "LingBot Batch",
                Path("configs/runs/lingbot"),
                partial(_validate_batch, root),
            ),
            ConfigKind(
                "reset",
                "A1 reset pose",
                Path("configs/poses"),
                partial(_validate_reset, root),
                include=looks_like_a1_pose,
            ),
        ),
    )


def looks_like_a1_pose(path: Path) -> bool:
    data = tomllib.loads(path.read_text())
    return set(data) == {"joints", "gripper", "motion"}


def _validate_teleop(root: Path, path: Path) -> None:
    load_teleop_config(path, repo_root=root)


def _validate_deployment(root: Path, path: Path) -> None:
    load_lingbot_config(path, repo_root=root)


def _validate_batch(root: Path, path: Path) -> None:
    candidate = load_lingbot_batch_config(path, repo_root=root)
    for existing_path in sorted((root / "configs/runs/lingbot").glob("*.toml")):
        if existing_path.resolve() == path.resolve():
            continue
        existing = load_lingbot_batch_config(existing_path, repo_root=root)
        if existing.batch_id == candidate.batch_id:
            raise ValueError(
                f"batch.id must be unique; {candidate.batch_id!r} is already "
                f"owned by {existing.path.name}"
            )


def _validate_reset(root: Path, path: Path) -> None:
    system = load_system_config(root / SYSTEM_CONFIG, repo_root=root)
    load_a1_home_pose(path, system=system, repo_root=root)
