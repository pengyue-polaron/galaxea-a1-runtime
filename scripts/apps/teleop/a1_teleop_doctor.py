#!/usr/bin/env python3
"""Configuration-driven static checks for the A1 SO100 teleop adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from galaxea_a1_runtime.runtime.health_checks import (  # noqa: E402
    Check,
    add_check,
    finish_checks,
)
from galaxea_a1_runtime.teleop.config import (  # noqa: E402
    default_config_path,
    load_teleop_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args(argv)
    config = load_teleop_config(args.config, repo_root=ROOT)
    checks: list[Check] = []
    add_check(
        checks,
        "leader_port",
        Path(config.leader.port).exists(),
        config.leader.port,
        required=True,
    )
    for name in (
        "lerobot.teleoperators.so_leader",
        "galaxea_a1_runtime.teleop.a1_so_leader",
        "rospy",
        "signal_arm",
    ):
        add_check(
            checks,
            f"{name.rsplit('.', maxsplit=1)[-1]}_import",
            importlib.util.find_spec(name) is not None,
            name,
            required=args.require_execution
            or not name.startswith(("rospy", "signal_arm")),
        )
    return finish_checks(checks, json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
