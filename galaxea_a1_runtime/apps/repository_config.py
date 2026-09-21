"""Agent-owned A1 configuration creation and validation."""

from __future__ import annotations

import tomllib
from functools import partial
from pathlib import Path

from embodied_ops.operator_panel import DocumentKind, RepositoryDocumentStore

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config, _load_backend
from galaxea_a1_runtime.models.config import load_model_config
from galaxea_a1_runtime.models.release import load_model_release
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.apps.tfp.model_config import load_tfp_model_config
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.teleop.config import load_teleop_config


def build_a1_document_store(repo_root: Path) -> RepositoryDocumentStore:
    root = repo_root.resolve()
    return RepositoryDocumentStore(
        root,
        (
            DocumentKind(
                kind_id="tfp-model",
                label="TFP transferred checkpoint",
                directory=Path("configs/models/tfp"),
                suffix=".checkpoint.toml",
                language="TOML",
                validate=lambda path: load_tfp_model_config(path, repo_root=root),
            ),
            DocumentKind(
                kind_id="inference-backend",
                label="LingBot-family inference backend",
                directory=Path("configs/inference/backends"),
                suffix=".toml",
                language="TOML",
                validate=lambda path: _load_backend(path, root),
                include=_is_lingbot_backend,
            ),
            DocumentKind(
                kind_id="diffusion2one-model",
                label="Diffusion2One model component",
                directory=Path("configs/models/diffusion2one"),
                suffix=".toml",
                language="TOML",
                validate=lambda path: load_model_config(path, repo_root=root),
                include=lambda path: not path.name.endswith(".contract.toml"),
            ),
            DocumentKind(
                kind_id="model-release",
                label="Multi-model release plan",
                directory=Path("configs/releases"),
                suffix=".toml",
                language="TOML",
                validate=lambda path: load_model_release(path, repo_root=root),
            ),
            DocumentKind(
                kind_id="diffusion2one-deployment",
                label="Diffusion2One deployment",
                directory=Path("configs/deployments/diffusion2one"),
                suffix=".toml",
                language="TOML",
                validate=partial(_validate_deployment, root),
            ),
            DocumentKind(
                kind_id="teleop",
                label="Teleop",
                directory=Path("configs/teleop"),
                suffix=".toml",
                language="TOML",
                validate=partial(_validate_teleop, root),
            ),
            DocumentKind(
                kind_id="deployment",
                label="LingBot deployment",
                directory=Path("configs/deployments/lingbot"),
                suffix=".toml",
                language="TOML",
                validate=partial(_validate_deployment, root),
            ),
            DocumentKind(
                kind_id="batch",
                label="LingBot Batch",
                directory=Path("configs/runs/lingbot"),
                suffix=".toml",
                language="TOML",
                validate=partial(_validate_batch, root),
            ),
            DocumentKind(
                kind_id="reset",
                label="A1 reset pose",
                directory=Path("configs/poses"),
                suffix=".toml",
                language="TOML",
                validate=partial(_validate_reset, root),
                include=looks_like_a1_pose,
            ),
        ),
    )


def looks_like_a1_pose(path: Path) -> bool:
    data = tomllib.loads(path.read_text())
    return set(data) == {"joints", "gripper", "motion"}


def _is_lingbot_backend(path: Path) -> bool:
    data = tomllib.loads(path.read_text())
    return data.get("backend", {}).get("adapter") in {"lingbot_va", "diffusion2one"}


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
