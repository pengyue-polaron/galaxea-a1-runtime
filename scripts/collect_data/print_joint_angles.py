#!/usr/bin/env python3
import argparse
import math
import os
import sys

# Let script run even if user uses a non-ROS python env.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import rospy
from sensor_msgs.msg import JointState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print clean A1 joint angles from ROS topic.")
    parser.add_argument("--topic", default="/joint_states_host")
    parser.add_argument("--count", type=int, default=0, help="0 means print forever")
    parser.add_argument("--unit", choices=("deg", "rad"), default="deg")
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    args = parse_args()
    scale = 180.0 / math.pi if args.unit == "deg" else 1.0
    suffix = args.unit
    printed = {"n": 0}

    print(f"Printing joint angles from {args.topic} ({suffix})")

    def cb(msg: JointState) -> None:
        names = list(msg.name) if msg.name else []
        parts = []
        for i, pos in enumerate(msg.position):
            name = names[i] if i < len(names) else f"joint{i + 1}"
            parts.append(f"{name}={pos * scale:8.3f}{suffix}")
        if parts:
            print(" | ".join(parts))
            sys.stdout.flush()
            printed["n"] += 1
        if args.count > 0 and printed["n"] >= args.count:
            rospy.signal_shutdown("done")

    rospy.init_node("a1_joint_angle_printer", anonymous=True)
    rospy.Subscriber(args.topic, JointState, cb, queue_size=10)
    rospy.spin()


if __name__ == "__main__":
    main()
