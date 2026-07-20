#!/usr/bin/env python3
# ruff: noqa: E402
"""Operator entrypoint for configuration-driven teleop recording."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.teleop.config import (
    default_config_path,
    load_teleop_config,
    validate_collection_config,
)
from galaxea_a1_runtime.console import ArgumentParser


def main() -> int:
    parser = ArgumentParser(description="Record Galaxea A1 teleop episodes.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task")
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    args = parser.parse_args()
    config = load_teleop_config(args.config, repo_root=ROOT)
    validate_collection_config(config)
    from galaxea_a1_runtime.apps.teleop.collector import run_safely

    return run_safely(config, experiment=args.experiment, task=args.task)


if __name__ == "__main__":
    raise SystemExit(main())
