"""Discover registered LingBot-family deployments as selectable targets."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from embodied_ops import TaskCatalog

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.configuration.base import repo_path
from galaxea_a1_runtime.models.config import ModelArtifactConfig

LINGBOT_FAMILY_ADAPTERS = frozenset({"lingbot_va", "diffusion2one"})


@dataclass(frozen=True)
class InferenceTarget:
    path: Path
    deployment_id: str
    adapter: str
    model: ModelArtifactConfig
    catalog: TaskCatalog
    execute: bool
    ready: bool
    artifact_state: str

    @property
    def key(self) -> str:
        return self.path.stem


def discover_targets(repo_root: Path) -> tuple[InferenceTarget, ...]:
    """Load every LingBot-family deployment; unrelated families are skipped."""

    root = repo_root.resolve()
    targets: list[InferenceTarget] = []
    for path in sorted((root / "configs/deployments").glob("**/*.toml")):
        if _backend_adapter(path, root) not in LINGBOT_FAMILY_ADAPTERS:
            continue
        config = load_lingbot_config(path, repo_root=root)
        data = tomllib.loads(path.read_text())
        deployment = data["deployment"]
        model = config.policy_server.model
        targets.append(
            InferenceTarget(
                path=path,
                deployment_id=str(deployment["id"]),
                adapter=config.policy_server.backend.adapter,
                model=model,
                catalog=config.task_catalog,
                execute=config.execution.execute,
                ready=config.policy_server.deployment_ready,
                artifact_state=_artifact_state(model),
            )
        )
    return tuple(targets)


def _backend_adapter(path: Path, root: Path) -> str:
    data = tomllib.loads(path.read_text())
    reference = data.get("backend")
    if not isinstance(reference, dict) or not isinstance(reference.get("config"), str):
        raise ValueError(f"deployment is missing [backend].config: {path}")
    backend_path = repo_path(root, reference["config"])
    if not backend_path.is_file():
        raise FileNotFoundError(f"deployment backend config is missing: {backend_path}")
    backend = tomllib.loads(backend_path.read_text()).get("backend")
    if not isinstance(backend, dict):
        raise ValueError(f"backend config is missing [backend]: {backend_path}")
    adapter = backend.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError(f"backend config is missing an adapter: {backend_path}")
    return adapter


def _artifact_state(model: ModelArtifactConfig) -> str:
    root = model.artifact_root
    if not root.is_dir():
        return "missing"
    for item in model.manifest.files:
        path = root / Path(*item.path.parts)
        if not path.is_file() or path.stat().st_size != item.size:
            return "incomplete"
    return "ready"


__all__ = ["InferenceTarget", "LINGBOT_FAMILY_ADAPTERS", "discover_targets"]
