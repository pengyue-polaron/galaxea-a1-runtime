#!/usr/bin/env python3
"""Static app checks for ACT joint-state deployment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    level: str
    detail: str


def add(checks: list[Check], name: str, ok: bool, detail: str, *, required: bool = True) -> None:
    checks.append(Check(name, "PASS" if ok else ("FAIL" if required else "WARN"), detail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wrist-backend", choices=("realsense", "v4l2"), required=True)
    parser.add_argument("--wrist-serial", default="")
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    checkpoint = Path(args.checkpoint)
    add(checks, "checkpoint", checkpoint.is_dir(), str(checkpoint), required=True)
    add(checks, "config_json", (checkpoint / "config.json").is_file(), str(checkpoint / "config.json"))
    add(checks, "model", (checkpoint / "model.safetensors").is_file(), str(checkpoint / "model.safetensors"))
    add(checks, "torch_import", importlib.util.find_spec("torch") is not None, "torch")
    add(checks, "lerobot_import", importlib.util.find_spec("lerobot") is not None, "lerobot")
    add(checks, "cv2_import", importlib.util.find_spec("cv2") is not None, "cv2")
    add(checks, "rospy_import", importlib.util.find_spec("rospy") is not None, "rospy", required=args.require_execution)
    if args.wrist_backend == "realsense":
        from galaxea_a1_runtime.hardware.cameras import realsense_device_info

        try:
            info = realsense_device_info(args.wrist_serial)
        except Exception as exc:
            add(checks, "wrist_camera", False, str(exc), required=True)
        else:
            add(checks, "wrist_camera", info is not None, str(info), required=True)
    elif args.wrist_camera == "auto":
        add(checks, "wrist_camera", True, "auto")
    else:
        wrist = Path(args.wrist_camera)
        add(checks, "wrist_camera", wrist.exists(), str(wrist), required=True)

    if args.json:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        width = max((len(item.name) for item in checks), default=0)
        for item in checks:
            print(f"[{item.level:4}] {item.name:<{width}}  {item.detail}")
    return 1 if any(item.level == "FAIL" for item in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
