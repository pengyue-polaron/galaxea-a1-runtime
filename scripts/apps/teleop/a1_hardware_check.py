#!/usr/bin/env python3
"""Thin entrypoint for the read-only A1 Teleop hardware check."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.teleop.hardware_check import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
