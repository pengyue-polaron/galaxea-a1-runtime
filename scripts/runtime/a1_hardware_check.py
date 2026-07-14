#!/usr/bin/env python3
"""Read-only hardware enumeration check for the A1 teleop rig."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.hardware.cameras import (  # noqa: E402
    realsense_device_info,
    realsense_usb_is_superspeed,
    resolve_video_source,
)
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config  # noqa: E402


DEFAULT_POSE_CONFIG = Path("configs/poses/a1_so100_collection_start.toml")


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Check enumerated A1 teleop hardware without moving it.")
    parser.add_argument("--config", type=Path, help="Teleop TOML config. Defaults to configs/teleop/a1_so100.toml")
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config_path = args.config or default_config_path(ROOT_DIR)
    config = load_teleop_config(config_path, repo_root=ROOT_DIR)
    pose_path = _resolve(args.pose_config, ROOT_DIR)
    pose = _read_toml(pose_path)
    checks: list[Check] = []

    a1_real = _check_device(checks, "a1_serial", config.host.a1_serial, required=True)
    leader_real = _check_device(checks, "leader_serial", config.leader.port, required=True)
    _add(
        checks,
        "leader_pose_config",
        _pose_leader_port(pose) == config.leader.port,
        f"teleop={config.leader.port}; pose={_pose_leader_port(pose) or '<missing>'}",
        required=True,
    )
    if a1_real and leader_real:
        _add(
            checks,
            "serial_distinct",
            a1_real != leader_real,
            f"a1={a1_real}; leader={leader_real}",
            required=True,
        )
    _check_busy(checks, "a1_serial_busy", config.host.a1_serial)
    _check_busy(checks, "leader_serial_busy", config.leader.port)

    _check_realsense(
        checks,
        name="realsense",
        serial=config.front_camera.serial,
        require_usb3=config.front_camera.require_usb3,
    )
    if config.wrist_camera.backend == "realsense":
        _check_realsense(
            checks,
            name="wrist_camera",
            serial=config.wrist_camera.serial,
            require_usb3=config.wrist_camera.require_usb3,
        )
    else:
        _check_wrist_camera(checks, config.wrist_camera.device)

    detail = f"teleop={config.path}; pose={pose_path}"
    _add(checks, "tracked_configs", True, detail, required=True)
    return _finish(checks, json_output=args.json)


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _pose_leader_port(pose: dict) -> str:
    leader = pose.get("leader", {})
    return str(leader.get("port", "")) if isinstance(leader, dict) else ""


def _check_device(checks: list[Check], name: str, device: str, *, required: bool) -> Path | None:
    path = Path(device)
    exists = path.exists()
    real = Path(os.path.realpath(path)) if exists else None
    detail = f"{device} -> {real}" if real else f"{device} missing"
    _add(checks, name, exists, detail, required=required)
    if not exists:
        return None
    access = os.access(path, os.R_OK | os.W_OK)
    _add(checks, f"{name}_access", access, "read/write ok" if access else "missing read/write permission", required=True)
    return real


def _check_busy(checks: list[Check], name: str, device: str) -> None:
    path = Path(device)
    if not path.exists():
        _add(checks, name, True, "skipped; device missing", required=False)
        return
    try:
        result = subprocess.run(
            ["fuser", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1.0,
        )
    except Exception as exc:
        _level(checks, name, "WARN", f"could not inspect users: {exc}")
        return
    pids = result.stdout.split()
    if pids:
        _level(checks, name, "WARN", "in use by " + "; ".join(_pid_detail(pid) for pid in pids))
    else:
        _add(checks, name, True, "not in use", required=False)


def _check_realsense(checks: list[Check], *, name: str, serial: str, require_usb3: bool) -> None:
    try:
        info = realsense_device_info(serial or None)
    except Exception as exc:
        _add(checks, name, False, str(exc), required=True)
        return
    if info is None:
        _add(checks, name, False, "no RealSense device enumerated", required=True)
        return
    serial_detail = f"{info.name} serial={info.serial or '<unknown>'} usb={info.usb_type or '<unknown>'}"
    _add(checks, name, True, serial_detail, required=True)
    usb_ok = (not require_usb3) or realsense_usb_is_superspeed(info.usb_type)
    detail = f"usb={info.usb_type or '<unknown>'}; require_usb3={require_usb3}"
    _add(checks, f"{name}_usb", usb_ok, detail, required=True)


def _check_wrist_camera(checks: list[Check], device: str) -> None:
    try:
        source, label = resolve_video_source(device)
    except Exception as exc:
        _add(checks, "wrist_camera", False, f"{device}: {exc}", required=True)
        return
    detail = f"{device} -> {source} ({label})"
    _add(checks, "wrist_camera", True, detail, required=True)


def _resolve(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def _pid_detail(pid: str) -> str:
    cmdline = Path("/proc") / pid / "cmdline"
    try:
        parts = [part for part in cmdline.read_text().split("\0") if part]
    except OSError:
        return f"pid={pid}"
    if not parts:
        return f"pid={pid}"
    command = Path(parts[0]).name
    if len(parts) > 1:
        command += " " + " ".join(parts[1:4])
    return f"pid={pid} ({command})"


def _add(checks: list[Check], name: str, ok: bool, detail: str, *, required: bool) -> None:
    checks.append(Check(name, "PASS" if ok else ("FAIL" if required else "WARN"), detail))


def _level(checks: list[Check], name: str, level: str, detail: str) -> None:
    checks.append(Check(name, level, detail))


def _finish(checks: list[Check], *, json_output: bool) -> int:
    if json_output:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        width = max((len(item.name) for item in checks), default=0)
        for item in checks:
            print(f"[{item.level:4}] {item.name:<{width}}  {item.detail}")
    return 1 if any(item.level == "FAIL" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
