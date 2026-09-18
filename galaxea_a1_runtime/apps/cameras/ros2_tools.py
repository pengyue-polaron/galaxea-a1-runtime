"""Read-only ROS 2 camera graph inspection and bounded MCAP recording."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from galaxea_a1_runtime.apps.cameras.ros2_pipeline import (
    CAMERA_NAMESPACE,
    ROS2_CAMERA_CONTAINER,
    ROS2_IMAGE,
    SYNC_DIAGNOSTICS_TOPIC,
    image_topic,
)
from galaxea_a1_runtime.configuration.system import SYSTEM_CONFIG, load_system_config
from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeReaders


def recording_topics(system) -> list[str]:
    topics = [SYNC_DIAGNOSTICS_TOPIC]
    for name in ("front", "wrist"):
        topics.extend(
            [
                image_topic(name),
                f"{CAMERA_NAMESPACE}/{name}/color/camera_info",
                f"{CAMERA_NAMESPACE}/{name}/color/metadata",
            ]
        )
    if system.cameras.front.depth:
        topics.append(f"{CAMERA_NAMESPACE}/front/aligned_depth_to_color/image_raw")
    return topics


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / SYSTEM_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    record = commands.add_parser("record")
    record.add_argument("duration_s", type=float)
    inspect = commands.add_parser("info")
    inspect.add_argument("bag", type=Path)
    args = parser.parse_args()
    system = load_system_config(args.config)
    if args.command == "status":
        subprocess.run(
            [
                "docker",
                "exec",
                ROS2_CAMERA_CONTAINER,
                "/ros_entrypoint.sh",
                "ros2",
                "topic",
                "list",
                "-t",
                "--no-daemon",
            ],
            check=True,
        )
        return subprocess.run(
            [
                "docker",
                "exec",
                ROS2_CAMERA_CONTAINER,
                "/ros_entrypoint.sh",
                "timeout",
                "5",
                "ros2",
                "topic",
                "echo",
                SYNC_DIAGNOSTICS_TOPIC,
                "std_msgs/msg/String",
                "--once",
                "--full-length",
            ],
            check=False,
        ).returncode
    if args.command == "info":
        bag = args.bag.resolve(strict=True)
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{bag}:/bag:ro",
                ROS2_IMAGE,
                "ros2",
                "bag",
                "info",
                "/bag",
            ],
            check=False,
        ).returncode
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration_s must be finite and positive")
    reader = CameraBridgeReaders(system.cameras)
    try:
        reader.start(timeout_s=system.web_preview.startup_timeout_s)
        pair = reader.latest_pair()
        if pair is None or any(
            time.perf_counter() - sample.monotonic_s > system.cameras.max_age_s
            for sample in pair
        ):
            raise RuntimeError("cannot record without a fresh synchronized camera pair")
    finally:
        reader.close()
    # Require the managed owner; recording never opens another physical device.
    subprocess.run(
        [
            "docker",
            "exec",
            ROS2_CAMERA_CONTAINER,
            "/ros_entrypoint.sh",
            "timeout",
            "5",
            "ros2",
            "topic",
            "echo",
            SYNC_DIAGNOSTICS_TOPIC,
            "std_msgs/msg/String",
            "--once",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    output_root = root / "outputs/ros2"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("cameras_%Y%m%dT%H%M%S_%fZ")
    container = f"a1-ros2-record-{os.getpid()}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--label",
        "io.galaxea.a1-runtime.managed=true",
        "--network",
        "host",
        "--ipc",
        "host",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{output_root}:/recordings",
        "-e",
        f"ROS_DOMAIN_ID={system.cameras.ros_domain_id}",
        "-e",
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        "-e",
        "ROS_LOG_DIR=/tmp/ros-log",
        ROS2_IMAGE,
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        f"/recordings/{run_id}",
        "--topics",
        *recording_topics(system),
    ]
    process = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"MCAP recorder exited early: {process.returncode}")
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run(
            ["docker", "kill", "--signal", "SIGINT", container], capture_output=True
        )
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "stop", "--time", "5", container], capture_output=True
            )
            process.wait(timeout=10)
    if (
        process.returncode != 0
        or not (output_root / run_id / "metadata.yaml").is_file()
    ):
        raise RuntimeError("MCAP recording did not finalize successfully")
    metadata_code = (
        "import json,rosbag2_py; "
        "m=rosbag2_py.Info().read_metadata('/bag','mcap'); "
        "print(json.dumps({t.topic_metadata.name:t.message_count "
        "for t in m.topics_with_message_count}))"
    )
    metadata = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{output_root / run_id}:/bag:ro",
            ROS2_IMAGE,
            "python3",
            "-c",
            metadata_code,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(metadata.stdout)
    if any(counts.get(image_topic(name), 0) == 0 for name in ("front", "wrist")):
        raise RuntimeError(
            f"MCAP capture has no images for one or both cameras; retained at {output_root / run_id}"
        )
    print(output_root / run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
