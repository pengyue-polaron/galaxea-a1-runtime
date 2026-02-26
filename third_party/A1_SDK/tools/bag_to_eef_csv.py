#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

# Allow running without manually sourcing ROS in common Ubuntu Noetic setup.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

try:
    import rosbag
except Exception as exc:
    print(
        f"Failed to import rosbag: {exc}\n"
        "Tip: source install/setup.bash (or /opt/ros/noetic/setup.bash) first.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EEF trajectory (and optional gripper trajectory) from a ROS bag to CSV."
    )
    parser.add_argument("--bag", required=True, help="Input .bag file path")
    parser.add_argument(
        "--topic",
        default="/end_effector_pose",
        help="Pose topic (default: /end_effector_pose)",
    )
    parser.add_argument(
        "--gripper-cmd-topic",
        default="/gripper_position_control_host",
        help="Gripper command topic (default: /gripper_position_control_host)",
    )
    parser.add_argument(
        "--gripper-feedback-topic",
        default="/gripper_stroke_host",
        help="Gripper feedback topic (default: /gripper_stroke_host)",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Export only EEF fields (legacy behavior).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <bag_basename>_eef.csv)",
    )
    return parser.parse_args()


def stamp_to_sec(stamp):
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def get_pose(msg):
    if hasattr(msg, "pose") and hasattr(msg.pose, "position") and hasattr(msg.pose, "orientation"):
        return msg.pose
    if hasattr(msg, "position") and hasattr(msg, "orientation"):
        return msg
    return None


def get_timestamp(msg, bag_time):
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        if stamp.secs != 0 or stamp.nsecs != 0:
            return stamp_to_sec(stamp)
    return stamp_to_sec(bag_time)


def get_gripper_cmd(msg) -> Optional[float]:
    if hasattr(msg, "gripper_stroke"):
        return float(msg.gripper_stroke)
    return None


def get_gripper_feedback(msg) -> Optional[float]:
    if hasattr(msg, "position") and len(msg.position) > 0:
        return float(msg.position[0])
    return None


def format_float_or_empty(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.9f}"


def main():
    args = parse_args()
    include_gripper = not args.no_gripper
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        print(f"Bag file not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        stem, _ = os.path.splitext(bag_path)
        out_path = f"{stem}_eef.csv"
    else:
        out_path = os.path.abspath(args.out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pose_rows: List[Tuple[float, object]] = []
    gripper_cmd_events: List[Tuple[float, float]] = []
    gripper_feedback_events: List[Tuple[float, float]] = []
    topics = [args.topic]
    if include_gripper:
        topics.extend([args.gripper_cmd_topic, args.gripper_feedback_topic])

    rows = 0
    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]
        if include_gripper:
            header.extend(["gripper_cmd", "gripper_feedback"])
        writer.writerow(header)

        for topic, msg, t in bag.read_messages(topics=topics):
            ts = get_timestamp(msg, t)
            if topic == args.topic:
                pose = get_pose(msg)
                if pose is None:
                    continue
                pose_rows.append((ts, pose))
                continue

            if not include_gripper:
                continue

            if topic == args.gripper_cmd_topic:
                gripper_cmd = get_gripper_cmd(msg)
                if gripper_cmd is not None:
                    gripper_cmd_events.append((ts, gripper_cmd))
                continue

            if topic == args.gripper_feedback_topic:
                gripper_feedback = get_gripper_feedback(msg)
                if gripper_feedback is not None:
                    gripper_feedback_events.append((ts, gripper_feedback))

        pose_rows.sort(key=lambda x: x[0])
        gripper_cmd_events.sort(key=lambda x: x[0])
        gripper_feedback_events.sort(key=lambda x: x[0])

        cmd_idx = -1
        fb_idx = -1
        for ts, pose in pose_rows:
            while cmd_idx + 1 < len(gripper_cmd_events) and gripper_cmd_events[cmd_idx + 1][0] <= ts:
                cmd_idx += 1
            while fb_idx + 1 < len(gripper_feedback_events) and gripper_feedback_events[fb_idx + 1][0] <= ts:
                fb_idx += 1

            row = [
                f"{ts:.9f}",
                f"{pose.position.x:.9f}",
                f"{pose.position.y:.9f}",
                f"{pose.position.z:.9f}",
                f"{pose.orientation.x:.9f}",
                f"{pose.orientation.y:.9f}",
                f"{pose.orientation.z:.9f}",
                f"{pose.orientation.w:.9f}",
            ]
            if include_gripper:
                cmd_val = gripper_cmd_events[cmd_idx][1] if cmd_idx >= 0 else None
                fb_val = gripper_feedback_events[fb_idx][1] if fb_idx >= 0 else None
                row.extend([format_float_or_empty(cmd_val), format_float_or_empty(fb_val)])
            writer.writerow(row)
            rows += 1

    print(f"Wrote {rows} rows to {out_path}")
    if rows == 0:
        print(
            f"Warning: no pose messages found on topic '{args.topic}'.",
            file=sys.stderr,
        )
    if include_gripper:
        if len(gripper_cmd_events) == 0:
            print(
                f"Warning: no gripper command messages found on topic '{args.gripper_cmd_topic}'.",
                file=sys.stderr,
            )
        if len(gripper_feedback_events) == 0:
            print(
                f"Warning: no gripper feedback messages found on topic '{args.gripper_feedback_topic}'.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
