"""Shared code-checkout and locked-environment backend boundary."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from galaxea_a1_runtime.configuration.base import (
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
    backend_id = _identifier(string(backend, "id"), label="backend.id")
    adapter = _identifier(string(backend, "adapter"), label="backend.adapter")
    repository = string(source, "repository")
    if not repository.startswith("https://"):
        raise ValueError("backend source.repository must use https://")
    revision = _hex_digest(string(source, "revision"), 40, label="source.revision")
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
            checkout=_repo_path(repo_root, string(source, "checkout")),
        ),
        environment=BackendEnvironmentConfig(
            manager=manager_text,
            python_version=python_version,
            python=_repo_path(repo_root, string(environment, "python")),
            lock=_repo_path(repo_root, string(environment, "lock")),
            lock_sha256=_hex_digest(
                string(environment, "lock_sha256"),
                64,
                label="environment.lock_sha256",
            ),
        ),
    )


def ensure_backend_checkout(config: CodeBackendConfig) -> None:
    checkout = config.source.checkout
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                config.source.repository,
                str(checkout),
            ]
        )
        _run(["git", "checkout", "--detach", config.source.revision], cwd=checkout)
    verify_backend_checkout(config)


def verify_backend_checkout(config: CodeBackendConfig) -> None:
    checkout = config.source.checkout
    if not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"backend checkout is missing: {checkout}")
    repository = _output(["git", "remote", "get-url", "origin"], cwd=checkout)
    if repository != config.source.repository:
        raise ValueError(
            "backend checkout origin mismatch: expected "
            f"{config.source.repository}, got {repository}"
        )
    revision = _output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if revision != config.source.revision:
        raise ValueError(
            f"backend checkout revision mismatch: expected {config.source.revision}, got {revision}"
        )
    dirty = _output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout
    )
    if dirty:
        raise ValueError(f"backend checkout has tracked modifications: {checkout}")
    verify_environment_lock(config)


def verify_environment_lock(config: CodeBackendConfig) -> None:
    lock = config.environment.lock
    if not lock.is_file():
        raise FileNotFoundError(f"backend environment lock is missing: {lock}")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if digest != config.environment.lock_sha256:
        raise ValueError(
            f"backend environment lock SHA256 mismatch: expected "
            f"{config.environment.lock_sha256}, got {digest}"
        )


def ensure_backend_environment(config: CodeBackendConfig) -> None:
    verify_backend_checkout(config)
    environment = config.environment
    if environment.manager == "requirements-lock":
        if not environment.python.is_file():
            environment.python.parent.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    "uv",
                    "venv",
                    "--python",
                    environment.python_version,
                    str(environment.python.parent.parent),
                ]
            )
        _run(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(environment.python),
                "--index-strategy",
                "unsafe-best-match",
                str(environment.lock),
            ],
            env={**os.environ, "UV_HTTP_TIMEOUT": "300"},
        )
        _run(["uv", "pip", "check", "--python", str(environment.python)])
        verify_backend_environment(config)
        return
    project_environment = environment.python.parent.parent
    _run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--project",
            str(config.source.checkout),
            "--python",
            environment.python_version,
        ],
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(project_environment),
            "UV_HTTP_TIMEOUT": "300",
        },
    )
    verify_backend_environment(config)


def verify_backend_environment(config: CodeBackendConfig) -> None:
    verify_backend_checkout(config)
    python = config.environment.python
    if not python.is_file():
        raise FileNotFoundError(f"backend Python is missing: {python}")
    version = _output(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    )
    if version != config.environment.python_version:
        raise ValueError(
            "backend Python version mismatch: expected "
            f"{config.environment.python_version}, got {version}"
        )


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(path))


def _identifier(value: str, *, label: str) -> str:
    if not value or any(
        not (character.isalnum() or character in {"-", "_", "."}) for character in value
    ):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")
    return value


def _hex_digest(value: str, length: int, *, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
