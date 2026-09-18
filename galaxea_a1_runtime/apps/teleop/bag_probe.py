"""Read-only delivery probe in the recorder's own ROS 2 namespace and user."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    import rclpy
    from rclpy.node import Node
    from rosidl_runtime_py.utilities import get_message

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("timeout", type=float)
    args = parser.parse_args()
    topics = json.loads(args.manifest.read_text())["topics"]
    seen = {}
    rclpy.init()
    node = Node(
        "a1_bag_delivery_probe", enable_rosout=False, start_parameter_services=False
    )
    subscriptions = []

    def received(role, message):
        seen[role] = {
            "source_ns": message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec,
            "receive_ns": time.time_ns(),
        }

    try:
        for role, topic in topics.items():
            typ = (
                "sensor_msgs/msg/Image"
                if role in ("front", "wrist", "depth")
                else "geometry_msgs/msg/PoseStamped"
                if role == "eef"
                else "sensor_msgs/msg/JointState"
            )
            subscriptions.append(
                node.create_subscription(
                    get_message(typ),
                    topic,
                    lambda message, role=role: received(role, message),
                    10,
                )
            )
        deadline = time.monotonic() + args.timeout
        while set(seen) != set(topics) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if set(seen) != set(topics):
            raise RuntimeError(
                f"recorder namespace has no delivery for {sorted(set(topics) - set(seen))}"
            )
        args.manifest.with_name("delivery.json").write_text(
            json.dumps(seen, indent=2) + "\n"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
