"""Small, reusable helpers for Git-tracked TOML configuration files."""

from __future__ import annotations

import math
import os
import shlex
import tomllib
from pathlib import Path
from typing import Any


def load_toml(
    path: Path, *, repo_root: Path | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    path = path.resolve()
    root = repo_root.resolve() if repo_root is not None else discover_repo_root(path)
    return path, root, tomllib.loads(path.read_text())


def discover_repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "galaxea_a1_runtime"
        ).is_dir():
            return candidate
    raise ValueError(f"cannot discover repository root from {path}")


def required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{key}] table")
    return value


def require_exact_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - data.keys())
    unknown = sorted(data.keys() - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"invalid {label} keys: {', '.join(details)}")


def string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string value: {key}")
    return value


def text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing text value: {key}")
    return value


def boolean(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"missing boolean value: {key}")
    return value


def integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing integer value: {key}")
    return value


def integer_tuple(
    data: dict[str, Any],
    key: str,
    expected_len: int | None = None,
    *,
    min_len: int = 0,
) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{key} must contain only integers")
    result = tuple(value)
    if len(result) < min_len:
        raise ValueError(f"{key} expects at least {min_len} values")
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(result)}")
    return result


def floating(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric value: {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def float_tuple(
    data: dict[str, Any], key: str, expected_len: int | None = None
) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a number list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{key} must contain only numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{key} must contain only finite numbers")
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"{key} expects {expected_len} values, got {len(result)}")
    return result


def string_tuple(data: dict[str, Any], key: str, expected_len: int) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a string list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must contain only strings")
    result = tuple(value)
    if len(result) != expected_len or any(not item for item in result):
        raise ValueError(f"{key} expects {expected_len} non-empty strings")
    return result


def identifier(value: str, *, label: str) -> str:
    if not value or any(
        not (character.isalnum() or character in {"-", "_", "."}) for character in value
    ):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")
    return value


def lower_identifier(value: str, *, label: str) -> str:
    if not value or any(
        not (character.islower() or character.isdigit() or character in {"-", "_"})
        for character in value
    ):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")
    return value


def hex_digest(value: str, length: int, *, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def absolute_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(path))


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two resolved file-tree locations contain one another."""

    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def referenced_config(
    data: dict[str, Any], repo_root: Path, *, key: str = "system"
) -> Path:
    reference = required_table(data, key)
    require_exact_keys(reference, required={"config"}, label=f"{key} reference")
    return repo_path(repo_root, string(reference, "config"))


def number(value: float) -> str:
    return f"{value:g}"


def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"
