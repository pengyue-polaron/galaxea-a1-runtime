"""Versioned local-process transport shared by Runtime service boundaries."""

from __future__ import annotations

import os
import socket
import stat
import struct
from pathlib import Path
from typing import Any

import msgpack

PACKET_LENGTH = struct.Struct("!I")


def process_state_root(*, state_root: Path | None = None) -> Path:
    """Return the private per-user directory for Runtime process endpoints."""

    if state_root is not None:
        return state_root.expanduser().resolve()
    configured = os.environ.get("A1_PROCESS_STATE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    return (runtime_root / f"galaxea-a1-runtime-{os.getuid()}").resolve()


def process_socket_path(name: str, *, state_root: Path | None = None) -> Path:
    """Return one socket below the private Runtime process-state directory."""

    if not name or Path(name).name != name:
        raise ValueError("process socket name must be one non-empty path component")
    return process_state_root(state_root=state_root) / name


def prepare_unix_socket(path: Path) -> None:
    """Create the private parent and remove only a proven stale socket."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"local IPC path exists and is not a socket: {path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(str(path))
    except OSError:
        path.unlink()
    else:
        raise RuntimeError(f"local IPC service is already listening: {path}")
    finally:
        probe.close()


def send_packet(active_socket: socket.socket, value: Any) -> None:
    """Send one length-prefixed MessagePack value."""

    payload = msgpack.packb(value, use_bin_type=True)
    active_socket.sendall(PACKET_LENGTH.pack(len(payload)) + payload)


def receive_packet(
    active_socket: socket.socket,
    *,
    max_bytes: int,
) -> Any | None:
    """Receive one bounded length-prefixed MessagePack value."""

    header = receive_exact(active_socket, PACKET_LENGTH.size)
    if header is None:
        return None
    (size,) = PACKET_LENGTH.unpack(header)
    if size <= 0 or size > max_bytes:
        raise ValueError(f"invalid local IPC packet size: {size}")
    payload = receive_exact(active_socket, size)
    if payload is None:
        raise ConnectionError("local IPC packet ended early")
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


def receive_exact(active_socket: socket.socket, size: int) -> bytes | None:
    """Receive exactly ``size`` bytes, or ``None`` before any byte arrives."""

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = active_socket.recv(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise ConnectionError("local IPC connection ended mid-packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
