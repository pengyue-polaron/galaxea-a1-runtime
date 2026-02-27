#!/usr/bin/env python3
import argparse
import os
import sys
import time

# Let script run even if user forgets to source ROS env.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

try:
    import rospy
    from signal_arm.msg import gripper_position_control
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source install/setup.bash (or /opt/ros/noetic/setup.bash) first.",
        file=sys.stderr,
    )
    sys.exit(1)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a short one-shot OPEN/CLOSE gripper position command."
    )
    parser.add_argument("action", choices=("open", "close"))
    parser.add_argument("--topic", default="/gripper_position_control_host")
    parser.add_argument("--min-stroke", type=float, default=0.0)
    parser.add_argument("--max-stroke", type=float, default=60.0)
    parser.add_argument("--open-stroke", type=float, default=60.0)
    parser.add_argument("--close-stroke", type=float, default=0.0)
    parser.add_argument(
        "--burst-count",
        type=int,
        default=8,
        help="How many immediate messages to send at the beginning.",
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=0.01,
        help="Seconds between burst messages.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.25,
        help="Continue publishing the target for this long before exiting.",
    )
    parser.add_argument(
        "--hold-rate",
        type=float,
        default=40.0,
        help="Publish rate used during --hold-seconds.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0.05,
        help="Wait briefly after creating the publisher before publishing.",
    )
    return parser


def publish_stroke(pub, stroke: float):
    cmd = gripper_position_control()
    cmd.header.stamp = rospy.Time.now()
    cmd.gripper_stroke = float(stroke)
    pub.publish(cmd)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args(rospy.myargv()[1:])

    if args.min_stroke > args.max_stroke:
        parser.error("--min-stroke must be <= --max-stroke")
    if args.burst_count <= 0:
        parser.error("--burst-count must be > 0")
    if args.burst_interval < 0:
        parser.error("--burst-interval must be >= 0")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be >= 0")
    if args.hold_rate <= 0:
        parser.error("--hold-rate must be > 0")
    if args.startup_delay < 0:
        parser.error("--startup-delay must be >= 0")

    args.open_stroke = clamp(args.open_stroke, args.min_stroke, args.max_stroke)
    args.close_stroke = clamp(args.close_stroke, args.min_stroke, args.max_stroke)
    target_stroke = args.open_stroke if args.action == "open" else args.close_stroke

    rospy.init_node(f"a1_gripper_{args.action}_command", anonymous=True)
    pub = rospy.Publisher(args.topic, gripper_position_control, queue_size=10)

    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    rospy.loginfo("Sending gripper %s command: %.2f mm", args.action.upper(), target_stroke)
    for _ in range(args.burst_count):
        if rospy.is_shutdown():
            return 0
        publish_stroke(pub, target_stroke)
        if args.burst_interval > 0:
            time.sleep(args.burst_interval)

    if args.hold_seconds > 0:
        rate = rospy.Rate(args.hold_rate)
        end_time = time.monotonic() + args.hold_seconds
        while not rospy.is_shutdown() and time.monotonic() < end_time:
            publish_stroke(pub, target_stroke)
            rate.sleep()

    return 0


if __name__ == "__main__":
    sys.exit(main())
