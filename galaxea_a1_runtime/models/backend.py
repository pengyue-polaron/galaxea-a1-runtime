"""Parse A1 backend config and compose generic code-environment workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from embodied_ops.code_environment import (
    ensure_code_checkout,
    ensure_code_environment,
    verify_code_checkout,
    verify_code_environment,
)

from galaxea_a1_runtime.configuration.base import (
    absolute_path,
    hex_digest,
    identifier,
    integer,
    require_exact_keys,
    string,
)


EnvironmentManager = Literal["requirements-lock", "uv-lock"]


@dataclass(frozen=True)
class CodeSourceConfig:
    repository: str
    revision: str
    checkout: Path


@dataclass(frozen=True)
class BackendEnvironmentConfig:
    manager: EnvironmentManager
    python_version: str
    python: Path
    lock: Path
    lock_sha256: str


@dataclass(frozen=True)
class CodeBackendConfig:
    backend_id: str
    adapter: str
    source: CodeSourceConfig
    environment: BackendEnvironmentConfig


def isolated_backend_pythonpath(repo_root: Path) -> str:
    """Expose the narrow first-party sources imported by policy servers."""

    root = repo_root.resolve()
    sources = (
        root,
        root / "external/embodied-ops/src",
    )
    missing = [str(path) for path in sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"isolated backend Python sources are missing: {missing}"
        )
    return os.pathsep.join(str(path) for path in sources)


def parse_code_backend(
    *,
    backend: dict,
    source: dict,
    environment: dict,
    repo_root: Path,
) -> CodeBackendConfig:
    require_exact_keys(
        backend, required={"schema_version", "id", "adapter"}, label="backend"
    )
    require_exact_keys(
        source, required={"repository", "revision", "checkout"}, label="backend source"
    )
    require_exact_keys(
        environment,
        required={"manager", "python_version", "python", "lock", "lock_sha256"},
        label="backend environment",
    )
    if integer(backend, "schema_version") != 1:
        raise ValueError("backend.schema_version must be 1")
    backend_id = identifier(string(backend, "id"), label="backend.id")
    adapter = identifier(string(backend, "adapter"), label="backend.adapter")
    repository = string(source, "repository")
    if not repository.startswith("https://"):
        raise ValueError("backend source.repository must use https://")
    revision = hex_digest(string(source, "revision"), 40, label="source.revision")
    manager_text = string(environment, "manager")
    if manager_text not in {"requirements-lock", "uv-lock"}:
        raise ValueError(f"unsupported backend environment manager: {manager_text!r}")
    python_version = string(environment, "python_version")
    if len(python_version.split(".")) != 2 or any(
        not part.isdigit() for part in python_version.split(".")
    ):
        raise ValueError("environment.python_version must be major.minor")
    return CodeBackendConfig(
        backend_id=backend_id,
        adapter=adapter,
        source=CodeSourceConfig(
            repository=repository,
            revision=revision,
            checkout=absolute_path(repo_root, string(source, "checkout")),
        ),
        environment=BackendEnvironmentConfig(
            manager=manager_text,
            python_version=python_version,
            python=absolute_path(repo_root, string(environment, "python")),
            lock=absolute_path(repo_root, string(environment, "lock")),
            lock_sha256=hex_digest(
                string(environment, "lock_sha256"),
                64,
                label="environment.lock_sha256",
            ),
        ),
    )


def ensure_backend_checkout(config: CodeBackendConfig) -> None:
    ensure_code_checkout(config)


def verify_backend_checkout(config: CodeBackendConfig) -> None:
    verify_code_checkout(config)


def ensure_backend_environment(config: CodeBackendConfig) -> None:
    ensure_code_environment(config)


def verify_backend_environment(config: CodeBackendConfig) -> None:
    verify_code_environment(config)
