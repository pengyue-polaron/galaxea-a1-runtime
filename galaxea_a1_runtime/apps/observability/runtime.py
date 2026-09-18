"""Start the ROS 2 observation side of the persistent managed stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.observability import legacy_observation_topics
from galaxea_a1_runtime.runtime.local_ipc import process_state_root
from galaxea_a1_runtime.runtime.ros2 import ROS1_BRIDGE_IMAGE, ROS2_IMAGE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bridge-container", required=True)
    parser.add_argument("--native-container", required=True)
    parser.add_argument("--foxglove-container", required=True)
    args = parser.parse_args()
    system = load_system_config(args.config)
    root = Path(__file__).resolve().parents[3]
    for image in (ROS2_IMAGE, ROS1_BRIDGE_IMAGE):
        subprocess.run(
            ["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL
        )
    state_root = process_state_root()
    state_root.mkdir(parents=True, exist_ok=True)
    common = [
        "docker",
        "run",
        "-d",
        "--init",
        "--network",
        "host",
        "--ipc",
        "host",
        "--label",
        "io.galaxea.a1-runtime.managed=true",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{root}:/workspace:ro",
        "-e",
        f"ROS_DOMAIN_ID={system.ros2.domain_id}",
        "-e",
        "ROS_LOG_DIR=/tmp/ros-log",
    ]
    jazzy = [
        "-e",
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        "-e",
        f"A1_SYSTEM_CONFIG_PATH=/workspace/{system.path.relative_to(root)}",
    ]
    created = []
    try:
        subprocess.run(
            [
                *common,
                "--name",
                args.bridge_container,
                "-e",
                "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
                "-e",
                "ROS_MASTER_URI=http://127.0.0.1:11311",
                "-e",
                "ROS_IP=127.0.0.1",
                ROS1_BRIDGE_IMAGE,
                *(
                    value
                    for entry in legacy_observation_topics(system)
                    for value in entry
                ),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        created.append(args.bridge_container)
        subprocess.run(
            [
                *common,
                *jazzy,
                "--name",
                args.native_container,
                "-v",
                f"{state_root}:/run/galaxea-a1-runtime:ro",
                "-e",
                "A1_PROCESS_STATE_ROOT=/run/galaxea-a1-runtime",
                ROS2_IMAGE,
                "python3",
                "-m",
                "galaxea_a1_runtime.apps.observability.native",
                "--config",
                f"/workspace/{system.path.relative_to(root)}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        created.append(args.native_container)
        subprocess.run(
            [
                *common,
                *jazzy,
                "--name",
                args.foxglove_container,
                ROS2_IMAGE,
                "ros2",
                "launch",
                "/workspace/scripts/apps/observability/foxglove.launch.py",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        created.append(args.foxglove_container)
    except BaseException:
        if created:
            subprocess.run(
                ["docker", "rm", "-f", *created], check=False, stdout=subprocess.DEVNULL
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
