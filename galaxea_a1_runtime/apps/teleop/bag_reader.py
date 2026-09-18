"""Network-isolated Jazzy rosbag2 reader and host-side stream transport."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import msgpack

from galaxea_a1_runtime.runtime.ros2 import ROS2_IMAGE


def read_bag(episode: Path, *, images: bool = False):
    """Decode with official ROS types in an offline container, never ROS discovery."""
    repo = Path(__file__).resolve().parents[3]
    name = f"a1-bag-reader-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{repo}:/workspace:ro",
        "-v",
        f"{episode.resolve()}:/episode:ro",
        "-e",
        "ROS_LOG_DIR=/tmp/ros-log",
        ROS2_IMAGE,
        "python3",
        "-m",
        "galaxea_a1_runtime.apps.teleop.bag_reader",
        "/episode",
    ]
    if images:
        command.append("--images")
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        assert process.stdout is not None
        yield from msgpack.Unpacker(
            process.stdout, raw=False, max_buffer_size=128 * 1024 * 1024
        )
        if process.wait() != 0:
            raise RuntimeError(f"offline rosbag2 reader failed for {episode}")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            process.wait(timeout=10)


def main():
    # Native middleware/storage logs must not share the binary protocol stream.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.episode / "episode.json").read_text())
    roles = {topic: role for role, topic in manifest["topics"].items()}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.episode / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic, role in roles.items():
        expected = (
            "sensor_msgs/msg/Image"
            if role in ("front", "wrist", "depth")
            else "geometry_msgs/msg/PoseStamped"
            if role == "eef"
            else "sensor_msgs/msg/JointState"
        )
        if types.get(topic) != expected:
            raise ValueError(
                f"missing/wrong bag topic type: {topic}: {types.get(topic)}, expected {expected}"
            )
    seq = defaultdict(int)
    while reader.has_next():
        topic, raw, received = reader.read_next()
        if topic not in roles:
            continue
        role = roles[topic]
        index = seq[role]
        seq[role] += 1
        is_image = role in ("front", "wrist", "depth")
        if args.images and not is_image:
            continue
        message = deserialize_message(raw, get_message(types[topic]))
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        item = {
            "role": role,
            "index": index,
            "source_ns": stamp,
            "receive_ns": int(received),
        }
        if is_image:
            item.update(
                height=message.height,
                width=message.width,
                step=message.step,
                encoding=message.encoding,
                bigendian=message.is_bigendian,
            )
            if args.images:
                item["data"] = bytes(message.data)
        elif role == "eef":
            p, q = message.pose.position, message.pose.orientation
            item["values"] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        else:
            item.update(names=list(message.name), values=list(message.position))
        output.write(msgpack.packb(item, use_bin_type=True))
    output.flush()


if __name__ == "__main__":
    main()
