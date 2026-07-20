"""Strict, backend-independent model artifact descriptors."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from galaxea_a1_runtime.configuration.base import (
    absolute_path,
    hex_digest,
    identifier,
    integer,
    load_toml,
    require_exact_keys,
    required_table,
    string,
)


ProviderKind = Literal["huggingface"]


@dataclass(frozen=True)
class ModelFile:
    path: PurePosixPath
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelArtifactManifest:
    path: Path
    model_id: str
    provider: ProviderKind
    repo_id: str
    revision: str
    files: tuple[ModelFile, ...]
    sha256: str


@dataclass(frozen=True)
class ModelSourceConfig:
    provider: ProviderKind
    repo_id: str
    revision: str
    revision_label: str


@dataclass(frozen=True)
class ModelArtifactConfig:
    path: Path
    repo_root: Path
    model_id: str
    backend: str
    artifact_format: str
    checkpoint_step: int
    contract: Path
    source: ModelSourceConfig
    manifest: ModelArtifactManifest

    @property
    def artifact_root(self) -> Path:
        relative = PurePosixPath(self.model_id)
        return Path(
            os.path.abspath(
                self.repo_root
                / "models"
                / "artifacts"
                / Path(*relative.parts)
                / self.source.revision
            )
        )


def load_model_config(
    path: Path, *, repo_root: Path | None = None
) -> ModelArtifactConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(data, required={"model", "source"}, label="model config")
    model = required_table(data, "model")
    source = required_table(data, "source")
    require_exact_keys(
        model,
        required={
            "schema_version",
            "id",
            "backend",
            "artifact_format",
            "checkpoint_step",
            "manifest",
            "contract",
        },
        label="model",
    )
    require_exact_keys(
        source,
        required={"provider", "repo_id", "revision", "revision_label"},
        label="model source",
    )
    if integer(model, "schema_version") != 1:
        raise ValueError("model.schema_version must be 1")
    model_id = _model_id(string(model, "id"))
    backend = identifier(string(model, "backend"), label="model.backend")
    artifact_format = identifier(
        string(model, "artifact_format"), label="model.artifact_format"
    )
    checkpoint_step = integer(model, "checkpoint_step")
    if checkpoint_step < 0:
        raise ValueError("model.checkpoint_step must be non-negative")
    provider = _provider(string(source, "provider"))
    repo_id = string(source, "repo_id")
    if len(repo_id.split("/")) != 2 or any(
        not part or part in {".", ".."} for part in repo_id.split("/")
    ):
        raise ValueError("source.repo_id must be a namespace/name Hugging Face id")
    revision = hex_digest(string(source, "revision"), 40, label="source.revision")
    revision_label = string(source, "revision_label")
    manifest_path = absolute_path(repo_root, string(model, "manifest"))
    contract = absolute_path(repo_root, string(model, "contract"))
    if not contract.is_file():
        raise FileNotFoundError(f"model contract is missing: {contract}")
    manifest = load_model_manifest(
        manifest_path,
        expected_model_id=model_id,
        expected_provider=provider,
        expected_repo_id=repo_id,
        expected_revision=revision,
    )
    return ModelArtifactConfig(
        path=path,
        repo_root=repo_root,
        model_id=model_id,
        backend=backend,
        artifact_format=artifact_format,
        checkpoint_step=checkpoint_step,
        contract=contract,
        source=ModelSourceConfig(
            provider=provider,
            repo_id=repo_id,
            revision=revision,
            revision_label=revision_label,
        ),
        manifest=manifest,
    )


def load_model_manifest(
    path: Path,
    *,
    expected_model_id: str,
    expected_provider: ProviderKind,
    expected_repo_id: str,
    expected_revision: str,
) -> ModelArtifactManifest:
    if not path.is_file():
        raise FileNotFoundError(f"model artifact manifest is missing: {path}")
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "model_id",
        "source",
        "files",
    }:
        raise ValueError(f"invalid model artifact manifest keys: {path}")
    if raw["schema_version"] != 1 or raw["model_id"] != expected_model_id:
        raise ValueError(f"model artifact manifest identity mismatch: {path}")
    source = raw["source"]
    if not isinstance(source, dict) or set(source) != {
        "provider",
        "repo_id",
        "revision",
    }:
        raise ValueError(f"invalid model artifact manifest source: {path}")
    if source != {
        "provider": expected_provider,
        "repo_id": expected_repo_id,
        "revision": expected_revision,
    }:
        raise ValueError(f"model artifact manifest provenance mismatch: {path}")
    raw_files = raw["files"]
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError(f"model artifact manifest files must be non-empty: {path}")
    files: list[ModelFile] = []
    for relative_text, expected in sorted(raw_files.items()):
        relative = _relative_artifact_path(relative_text)
        if not isinstance(expected, dict) or set(expected) != {"size", "sha256"}:
            raise ValueError(f"invalid manifest file entry: {relative_text}")
        size = expected["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid manifest file size: {relative_text}")
        digest = expected["sha256"]
        if not isinstance(digest, str):
            raise ValueError(f"invalid manifest file SHA256: {relative_text}")
        files.append(
            ModelFile(
                path=relative,
                size=size,
                sha256=hex_digest(
                    digest, 64, label=f"manifest SHA256 for {relative_text}"
                ),
            )
        )
    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return ModelArtifactManifest(
        path=path,
        model_id=expected_model_id,
        provider=expected_provider,
        repo_id=expected_repo_id,
        revision=expected_revision,
        files=tuple(files),
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _relative_artifact_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid artifact path: {value!r}")
    if path.parts[0] == ".cache":
        raise ValueError("artifact manifests must not include downloader cache files")
    return path


def _model_id(value: str) -> str:
    if "\\" in value:
        raise ValueError("model.id must use '/' separators")
    path = PurePosixPath(value)
    if len(path.parts) < 2 or any(
        part in {"", ".", ".."} or not _valid_identifier(part) for part in path.parts
    ):
        raise ValueError(
            "model.id must contain at least family/name safe path segments"
        )
    return value


def _valid_identifier(value: str) -> bool:
    return bool(value) and all(
        character.isalnum() or character in {"-", "_", "."} for character in value
    )


def _provider(value: str) -> ProviderKind:
    if value != "huggingface":
        raise ValueError(f"unsupported model source provider: {value!r}")
    return value
