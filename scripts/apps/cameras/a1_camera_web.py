#!/usr/bin/env python3
"""Start the ROS 2 camera pipeline and its isolated-consumer boundary."""

from galaxea_a1_runtime.apps.cameras.ros2_runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
