"""Shared Python-path bootstrap for the isolated ROS1/A1 SDK environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROS1_PYTHON_LOG_CONFIG = Path("scripts/runtime/ros1_python_logging.conf")


def configure_ros1_python(
    repo_root: Path,
    *,
    include_system_site: bool = True,
    remove_ros2: bool = False,
) -> None:
    """Expose ROS1 and A1 messages before importing rospy-dependent modules."""

    repo_root = repo_root.resolve()
    configured_log_path = os.environ.get("ROS_PYTHON_LOG_CONFIG_FILE")
    log_config = (
        Path(configured_log_path)
        if configured_log_path
        else repo_root / ROS1_PYTHON_LOG_CONFIG
    )
    if not log_config.is_file():
        raise FileNotFoundError(
            f"ROS1 Python logging configuration not found: {log_config}"
        )
    if configured_log_path is None:
        os.environ["ROS_PYTHON_LOG_CONFIG_FILE"] = str(log_config)

    if remove_ros2:
        sys.path[:] = [path for path in sys.path if "/opt/ros/humble" not in path]
    candidates = ["/opt/ros/noetic/lib/python3/dist-packages"]
    if include_system_site:
        candidates.append("/usr/lib/python3/dist-packages")
    candidates.extend(
        (
            str(
                repo_root
                / "third_party"
                / "A1_SDK"
                / "install"
                / "lib"
                / "python3"
                / "dist-packages"
            ),
            str(repo_root / ".cache" / "ros1_python_overlay"),
        )
    )
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)
