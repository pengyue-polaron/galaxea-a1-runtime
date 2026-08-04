"""Validation for the A1 Runtime's local transport endpoint."""

from __future__ import annotations

from pathlib import Path

from .contracts import RuntimeContractError

UNIX_PREFIX = "unix://"


def unix_socket_path(endpoint: str) -> Path:
    if not isinstance(endpoint, str) or endpoint.strip() != endpoint:
        raise RuntimeContractError(f"invalid A1 Runtime endpoint: {endpoint!r}")
    if not endpoint.startswith("unix:///"):
        raise RuntimeContractError("A1 Runtime endpoint must use an absolute unix:/// path")
    path = Path(endpoint.removeprefix(UNIX_PREFIX))
    if not path.is_absolute() or str(path) == "/":
        raise RuntimeContractError("A1 Runtime Unix socket path must be absolute and non-root")
    if len(str(path).encode()) > 100:
        raise RuntimeContractError(
            "A1 Runtime Unix socket path is too long for portable AF_UNIX use"
        )
    return path
