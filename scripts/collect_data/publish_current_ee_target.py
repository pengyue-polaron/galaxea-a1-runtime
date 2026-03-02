#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _extend_ros_python_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
        str(repo_root / "third_party" / "A1_SDK" / "install" / "lib" / "python3" / "dist-packages"),
    ]
    a1_sdk_root = os.environ.get("A1_SDK_ROOT")
    if a1_sdk_root:
        candidates.append(str(Path(a1_sdk_root) / "install" / "lib" / "python3" / "dist-packages"))
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


_extend_ros_python_paths()

try:
    import rospy
    from geometry_msgs.msg import PoseStamped
    from roslib.message import get_message_class
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source /opt/ros/noetic/setup.bash and third_party/A1_SDK/install/setup.bash first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Capture current end-effector pose and publish it to ee target topic."
    )
    parser.add_argument("--pose-topic", default="/end_effector_pose")
    parser.add_argument("--target-topic", default="/a1_ee_target")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--topic-timeout", type=float, default=25.0)
    parser.add_argument("--message-timeout", type=float, default=25.0)
    parser.add_argument("--connect-timeout", type=float, default=1.0)
    parser.add_argument("--publish-rate", type=float, default=30.0)
    parser.add_argument("--publish-count", type=int, default=15)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only wait for and print current pose; do not publish target.",
    )
    return parser.parse_args(argv)


def wait_for_topic_type(topic: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    while not rospy.is_shutdown():
        try:
            published = rospy.get_published_topics()
        except Exception:
            published = []
        for name, type_name in published:
            if name == topic:
                return type_name
        if deadline is not None and time.monotonic() > deadline:
            break
        time.sleep(0.1)
    raise RuntimeError(f"Topic not found or not published: {topic}")


def to_pose_stamped(msg, default_frame_id: str) -> PoseStamped:
    out = PoseStamped()
    out.header.stamp = rospy.Time.now()

    if hasattr(msg, "header") and hasattr(msg, "pose"):
        out.pose = msg.pose
        incoming_frame = getattr(msg.header, "frame_id", "")
        out.header.frame_id = incoming_frame or default_frame_id
        return out

    if hasattr(msg, "position") and hasattr(msg, "orientation"):
        out.pose = msg
        out.header.frame_id = default_frame_id
        return out

    raise RuntimeError(
        "Unsupported pose message type on source topic. Expected PoseStamped or Pose-compatible fields."
    )


def wait_for_subscriber(pub, timeout_s: float) -> bool:
    if timeout_s <= 0:
        return pub.get_num_connections() > 0
    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if pub.get_num_connections() > 0:
            return True
        time.sleep(0.05)
    return pub.get_num_connections() > 0


def wait_for_pose_message(topic: str, msg_cls, timeout_s: float):
    if timeout_s <= 0:
        return rospy.wait_for_message(topic, msg_cls)

    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return rospy.wait_for_message(topic, msg_cls, timeout=min(1.0, remaining))
        except rospy.ROSException:
            continue
    raise RuntimeError(
        f"Timed out waiting for message on {topic}. "
        "Please ensure ee-tracker is running and /joint_states_host is publishing."
    )


def main() -> int:
    args = parse_args(rospy.myargv()[1:])
    rospy.init_node("publish_current_ee_target", anonymous=True)

    if args.publish_rate <= 0:
        raise RuntimeError("--publish-rate must be > 0")
    if args.publish_count <= 0:
        raise RuntimeError("--publish-count must be > 0")

    topic_type = wait_for_topic_type(args.pose_topic, args.topic_timeout)
    msg_cls = get_message_class(topic_type)
    if msg_cls is None:
        raise RuntimeError(f"Could not resolve ROS message class for topic type: {topic_type}")

    msg = wait_for_pose_message(args.pose_topic, msg_cls, args.message_timeout)
    pose_msg = to_pose_stamped(msg, args.frame_id)

    p = pose_msg.pose.position
    q = pose_msg.pose.orientation
    if args.probe_only:
        print(f"Current EE pose is available on {args.pose_topic}")
        print(
            f"position=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), "
            f"quaternion=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
        )
        return 0

    pub = rospy.Publisher(args.target_topic, PoseStamped, queue_size=1)
    if not wait_for_subscriber(pub, args.connect_timeout):
        rospy.logwarn(
            "No subscriber on %s yet; still publishing for best effort.",
            args.target_topic,
        )

    rate = rospy.Rate(args.publish_rate)
    for _ in range(args.publish_count):
        pose_msg.header.stamp = rospy.Time.now()
        pub.publish(pose_msg)
        rate.sleep()

    print(f"Published current EE pose to {args.target_topic}")
    print(
        f"position=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), "
        f"quaternion=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
