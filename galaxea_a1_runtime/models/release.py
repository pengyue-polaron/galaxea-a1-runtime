"""Strict tracked release plans for one pinned multi-model family.

A release plan is the create-once registration input for a family of models
that share one Hugging Face revision, folder layout, and runtime class. It does
not own runtime values: generated deployments keep owning execution, ports, and
recording, while descriptors stay the authority for a single model identity.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from embodied_ops import TaskCatalog, load_task_catalog

from galaxea_a1_runtime.configuration.base import (
    identifier,
    integer,
    load_toml,
    lower_identifier,
    require_exact_keys,
    required_table,
    string,
)


@dataclass(frozen=True)
class ReleaseKind:
    name: str
    backend_id: str
    deployment_template: str
    contract_template: str


RELEASE_KINDS: dict[str, ReleaseKind] = {
    "student": ReleaseKind(
        name="student",
        backend_id="diffusion2one",
        deployment_template="configs/deployments/diffusion2one/fruit_blocks_eef.toml",
        contract_template="configs/models/diffusion2one/fruit_blocks_eef.contract.toml",
    ),
    "teacher": ReleaseKind(
        name="teacher",
        backend_id="diffusion2one_teacher",
        deployment_template=(
            "configs/deployments/diffusion2one/fruit_blocks_teacher_eef.toml"
        ),
        contract_template=(
            "configs/models/diffusion2one/fruit_blocks_teacher_eef.contract.toml"
        ),
    ),
}

# Every registered model directory carries this exact release file set. A
# layout change must be reviewed here instead of silently widening a manifest.
RELEASE_MODEL_FILES: tuple[str, ...] = (
    "README.md",
    "eef.json",
    "lingbot_norm_stat.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
)

_MIN_PORT = 1024
_MAX_PORT = 65535


@dataclass(frozen=True)
class ReleaseSource:
    provider: str
    repo_id: str
    revision: str
    revision_label: str


@dataclass(frozen=True)
class ReleaseModel:
    key: str
    model_id: str
    directory: PurePosixPath
    checkpoint_step: int
    kind: ReleaseKind
    catalog: PurePosixPath
    catalog_record: TaskCatalog


@dataclass(frozen=True)
class ModelRelease:
    path: Path
    repo_root: Path
    release_id: str
    source: ReleaseSource
    server_port_base: int
    master_port_base: int
    models: tuple[ReleaseModel, ...]

    def manifest_path(self, model: ReleaseModel) -> Path:
        return (
            self.repo_root
            / "configs/models/diffusion2one"
            / (model.key + ".manifest.json")
        )

    def contract_path(self, model: ReleaseModel) -> Path:
        return (
            self.repo_root
            / "configs/models/diffusion2one"
            / (model.key + ".contract.toml")
        )

    def descriptor_path(self, model: ReleaseModel) -> Path:
        return self.repo_root / "configs/models/diffusion2one" / (model.key + ".toml")

    def deployment_path(self, model: ReleaseModel) -> Path:
        return (
            self.repo_root / "configs/deployments/diffusion2one" / (model.key + ".toml")
        )

    def deployment_id(self, model: ReleaseModel) -> str:
        return "diffusion2one-" + model.key.replace("_", "-")

    def server_port(self, index: int) -> int:
        return self.server_port_base + index

    def master_port(self, index: int) -> int:
        return self.master_port_base + index


def load_model_release(path: Path, *, repo_root: Path | None = None) -> ModelRelease:
    path, root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"release", "source", "ports", "model"},
        label="model release plan",
    )
    release = required_table(data, "release")
    require_exact_keys(release, required={"schema_version", "id"}, label="release")
    if integer(release, "schema_version") != 1:
        raise ValueError("release.schema_version must be 1")
    release_id = identifier(string(release, "id"), label="release.id")

    source = required_table(data, "source")
    require_exact_keys(
        source,
        required={"provider", "repo_id", "revision", "revision_label"},
        label="release source",
    )
    provider = string(source, "provider")
    if provider != "huggingface":
        raise ValueError(f"unsupported release source provider: {provider!r}")
    repo_id = string(source, "repo_id")
    if len(repo_id.split("/")) != 2 or any(
        not part or part in {".", ".."} for part in repo_id.split("/")
    ):
        raise ValueError("source.repo_id must be a namespace/name Hugging Face id")
    revision = string(source, "revision")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source.revision must be a 40-character lowercase hex digest")

    ports = required_table(data, "ports")
    require_exact_keys(ports, required={"server", "master"}, label="release ports")
    server_base = _port(ports, "server")
    master_base = _port(ports, "master")

    entries = data["model"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("a model release plan requires at least one [[model]] entry")

    models: list[ReleaseModel] = []
    keys: set[str] = set()
    model_ids: set[str] = set()
    directories: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each [[model]] entry must be a table")
        require_exact_keys(
            entry,
            required={"key", "id", "directory", "checkpoint_step", "kind", "catalog"},
            label="release model",
        )
        key = lower_identifier(string(entry, "key"), label="model.key")
        if key in keys:
            raise ValueError(f"duplicate model.key in release plan: {key}")
        keys.add(key)
        model_id = _model_id(string(entry, "id"))
        if model_id in model_ids:
            raise ValueError(f"duplicate model.id in release plan: {model_id}")
        model_ids.add(model_id)
        directory = _directory(string(entry, "directory"))
        if directory.as_posix() in directories:
            raise ValueError(f"duplicate model.directory in release plan: {directory}")
        directories.add(directory.as_posix())
        checkpoint_step = integer(entry, "checkpoint_step")
        if checkpoint_step < 0:
            raise ValueError("model.checkpoint_step must be non-negative")
        kind_name = string(entry, "kind")
        if kind_name not in RELEASE_KINDS:
            allowed = ", ".join(sorted(RELEASE_KINDS))
            raise ValueError(f"unknown model.kind {kind_name!r}; expected: {allowed}")
        catalog_path = _catalog(root, string(entry, "catalog"))
        catalog = load_task_catalog(catalog_path, repo_root=root)
        if not catalog.tasks:
            raise ValueError(f"release catalog has no prompts: {catalog_path}")
        models.append(
            ReleaseModel(
                key=key,
                model_id=model_id,
                directory=directory,
                checkpoint_step=checkpoint_step,
                kind=RELEASE_KINDS[kind_name],
                catalog=PurePosixPath(catalog_path.relative_to(root).as_posix()),
                catalog_record=catalog,
            )
        )

    plan = ModelRelease(
        path=path,
        repo_root=root,
        release_id=release_id,
        source=ReleaseSource(
            provider=provider,
            repo_id=repo_id,
            revision=revision,
            revision_label=string(source, "revision_label"),
        ),
        server_port_base=server_base,
        master_port_base=master_base,
        models=tuple(models),
    )
    _reject_tracked_port_collisions(plan)
    return plan


def _port(data: dict[str, Any], key: str) -> int:
    value = integer(data, key)
    if not _MIN_PORT <= value <= _MAX_PORT:
        raise ValueError(f"ports.{key} must be between {_MIN_PORT} and {_MAX_PORT}")
    return value


def _model_id(value: str) -> str:
    if "\\" in value:
        raise ValueError("model.id must use '/' separators")
    path = PurePosixPath(value)
    if len(path.parts) < 2 or any(
        part in {"", ".", ".."} or not _safe_segment(part) for part in path.parts
    ):
        raise ValueError(
            "model.id must contain at least family/name safe path segments"
        )
    return value


def _safe_segment(value: str) -> bool:
    return bool(value) and all(
        character.isalnum() or character in {"-", "_", "."} for character in value
    )


def _directory(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("model.directory must use '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", "..", ".cache"} for part in path.parts
    ):
        raise ValueError("model.directory must be a safe relative release folder")
    return path


def _catalog(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError("model.catalog must use '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("model.catalog must be a safe relative path")
    resolved = root.joinpath(*path.parts)
    if not resolved.is_file():
        raise FileNotFoundError(f"release catalog is missing: {resolved}")
    return resolved


def _reject_tracked_port_collisions(plan: ModelRelease) -> None:
    """Fail closed when generated ports overlap other tracked deployments."""

    generated = set()
    for index, model in enumerate(plan.models):
        generated.add(plan.server_port(index))
        generated.add(plan.master_port(index))
    own_paths = {plan.deployment_path(model) for model in plan.models}
    tracked: set[int] = set()
    deployment_root = plan.repo_root / "configs/deployments"
    if deployment_root.is_dir():
        for candidate in sorted(deployment_root.glob("**/*.toml")):
            if candidate in own_paths:
                continue
            try:
                data = tomllib.loads(candidate.read_text())
            except (OSError, tomllib.TOMLDecodeError):
                continue
            for table_name, key in (("server", "port"), ("session", "master_port")):
                table = data.get(table_name)
                if isinstance(table, dict):
                    value = table.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        tracked.add(value)
    collisions = sorted(generated & tracked)
    if collisions:
        raise ValueError(
            "release plan ports collide with tracked deployments: "
            + ", ".join(str(value) for value in collisions)
        )


__all__ = [
    "ModelRelease",
    "RELEASE_KINDS",
    "RELEASE_MODEL_FILES",
    "ReleaseKind",
    "ReleaseModel",
    "ReleaseSource",
    "load_model_release",
]
