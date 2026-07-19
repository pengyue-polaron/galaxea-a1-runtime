#!/usr/bin/env python3
# ruff: noqa: E402
"""Host entrypoint for the shared A1-only reset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT, remove_ros2=True)

from galaxea_a1_runtime.apps.reset.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
