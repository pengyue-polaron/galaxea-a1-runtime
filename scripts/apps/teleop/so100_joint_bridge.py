#!/usr/bin/env python3
# ruff: noqa: E402
"""Operator entrypoint for the SO leader joint bridge."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config
from galaxea_a1_runtime.console import ArgumentParser, info


def main() -> int:
    parser = ArgumentParser(
        description="SO leader -> staged A1 joint teleoperation bridge"
    )
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    args = parser.parse_args()
    config = load_teleop_config(args.config, repo_root=ROOT)
    from galaxea_a1_runtime.apps.teleop.bridge import run

    info(f"Teleop config: {config.path}")
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
