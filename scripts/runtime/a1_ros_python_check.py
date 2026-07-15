#!/usr/bin/env python3.12
"""Statically verify the tracked System config and ROS1 Python import boundary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.configuration.system import (  # noqa: E402
    DEFAULT_SYSTEM_CONFIG,
    load_system_config,
)
from galaxea_a1_runtime.console import ArgumentParser, success  # noqa: E402
from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python  # noqa: E402


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_SYSTEM_CONFIG)
    args = parser.parse_args()
    system = load_system_config(args.config, repo_root=ROOT)

    configure_ros1_python(ROOT)
    import rospy
    from sensor_msgs.msg import JointState
    from signal_arm.msg import arm_control

    success(
        f"ROS1 Python {sys.version_info.major}.{sys.version_info.minor} import ready; "
        f"config={system.path}; modules={rospy.__name__},"
        f"{JointState.__name__},{arm_control.__name__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
