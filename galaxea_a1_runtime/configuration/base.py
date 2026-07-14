"""Small, reusable helpers for Git-tracked TOML configuration files."""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path, *, repo_root: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    path = path.resolve()
    root = repo_root.resolve() if repo_root is not None else discover_repo_root(path)
    return path, root, tomllib.loads(path.read_text())


def discover_repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "galaxea_a1_runtime").is_dir():
            return candidate
    raise ValueError(f"cannot discover repository root from {path}")


def required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{key}] table")
    return value


def string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string value: {key}")
    return value


def float_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    result = tuple(float(item) for item in value)
    if len(result) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(result)}")
    return result


def string_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a string list")
    result = tuple(str(item) for item in value)
    if len(result) != expected_len or any(not item for item in result):
        raise ValueError(f"{key} expects {expected_len} non-empty strings")
    return result


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def referenced_config(data: dict[str, Any], repo_root: Path) -> Path:
    return repo_path(repo_root, string(required_table(data, "system"), "config"))


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def number(value: float) -> str:
    return f"{value:g}"


def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def shell_array(name: str, values: list[str]) -> str:
    return f"{name}=({' '.join(shlex.quote(value) for value in values)})"
