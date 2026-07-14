#!/usr/bin/env python3
# ruff: noqa: E402
"""Operator entrypoint for interactive teleoperation recording."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.teleop.collector import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
