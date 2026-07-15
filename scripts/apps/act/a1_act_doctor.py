#!/usr/bin/env python3
"""Configuration-driven static checks for ACT joint deployment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.act.config import (  # noqa: E402
    default_config_path,
    load_act_config,
)
from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from galaxea_a1_runtime.hardware.cameras import realsense_device_info  # noqa: E402
from galaxea_a1_runtime.runtime.health_checks import (  # noqa: E402
    Check,
    add_check,
    finish_checks,
)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args()
    config = load_act_config(args.config, repo_root=ROOT)
    checkpoint = config.policy.checkpoint
    checks: list[Check] = []
    add_check(checks, "checkpoint", checkpoint.is_dir(), str(checkpoint))
    add_check(
        checks,
        "config_json",
        (checkpoint / "config.json").is_file(),
        str(checkpoint / "config.json"),
    )
    add_check(
        checks,
        "model",
        (checkpoint / "model.safetensors").is_file(),
        str(checkpoint / "model.safetensors"),
    )
    for module in ("torch", "lerobot", "cv2"):
        add_check(
            checks,
            f"{module}_import",
            importlib.util.find_spec(module) is not None,
            module,
        )
    add_check(
        checks,
        "rospy_import",
        importlib.util.find_spec("rospy") is not None,
        "rospy",
        required=args.require_execution,
    )
    wrist = config.system.cameras.wrist
    if wrist.backend == "realsense":
        try:
            info = realsense_device_info(wrist.serial)
        except Exception as exc:
            add_check(checks, "wrist_camera", False, str(exc))
        else:
            add_check(checks, "wrist_camera", info is not None, str(info))
    elif wrist.device == "auto":
        add_check(checks, "wrist_camera", True, "auto")
    else:
        add_check(checks, "wrist_camera", Path(wrist.device).exists(), wrist.device)
    return finish_checks(checks, json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
