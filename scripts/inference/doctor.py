#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "cv2",
    "numpy",
    "openpi",
    "scipy",
    "torch",
    "tyro",
    "zmq",
)

OPTIONAL_MODULES = (
    "lerobot",
    "websockets",
)


def check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    openpi_dir = repo_root / "third_party" / "openpi"
    uv_path = shutil.which("uv")

    print(f"repo_root: {repo_root}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"uv: {uv_path or 'missing'}")
    print(f"third_party/openpi: {'present' if openpi_dir.is_dir() else 'missing'}")
    print(f"ROS setup sourced: {'yes' if os.environ.get('ROS_DISTRO') else 'no'}")
    print()

    missing_required = [name for name in REQUIRED_MODULES if not check_module(name)]
    missing_optional = [name for name in OPTIONAL_MODULES if not check_module(name)]

    print("required modules:")
    for name in REQUIRED_MODULES:
        status = "present" if name not in missing_required else "missing"
        print(f"  - {name}: {status}")

    print("optional modules:")
    for name in OPTIONAL_MODULES:
        status = "present" if name not in missing_optional else "missing"
        print(f"  - {name}: {status}")

    print("checks:")
    python_ok = sys.version_info >= (3, 11)
    print(f"  - python>=3.11: {'yes' if python_ok else 'no'}")
    print(f"  - infer env ready: {'yes' if python_ok and not missing_required and openpi_dir.is_dir() else 'no'}")

    if not python_ok:
        print("hint: create a dedicated infer env with Python 3.11 before installing OpenPI.")
    if not openpi_dir.is_dir():
        print("hint: add the missing `third_party/openpi` checkout before running infer.")
    if not uv_path:
        print("hint: install uv or activate the env where uv is available.")
    if missing_required:
        print("hint: install the missing required packages in your infer environment.")

    return 0 if python_ok and not missing_required and openpi_dir.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
