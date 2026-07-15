#!/usr/bin/env python3
# ruff: noqa: E402
"""Read-only side-by-side Teleop leader target and A1 gripper feedback."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.console import ArgumentParser, failure, info
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        parser.error("--rate-hz must be finite and positive")

    config = load_teleop_config(args.config, repo_root=ROOT)

    from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

    configure_ros1_python(ROOT)

    import rospy
    from sensor_msgs.msg import JointState
    from signal_arm.msg import gripper_position_control
    from std_msgs.msg import String

    from galaxea_a1_runtime.apps.teleop.gripper_debug import (
        build_gripper_debug_reading,
        format_gripper_debug_reading,
    )
    from galaxea_a1_runtime.hardware.freshness import LatestMessageCache
    from galaxea_a1_runtime.runtime.relay import RelayMonitor

    target = LatestMessageCache[float]()
    feedback = LatestMessageCache[float]()
    relay = RelayMonitor(config.system.relay.max_status_age_s)
    topics = config.system.topics

    def target_cb(msg: gripper_position_control) -> None:
        target.set(float(msg.gripper_stroke))

    def feedback_cb(msg: JointState) -> None:
        if msg.position:
            feedback.set(float(msg.position[0]))

    rospy.init_node("a1_gripper_debug", anonymous=False)
    rospy.Subscriber(
        topics.gripper_target, gripper_position_control, target_cb, queue_size=1
    )
    rospy.Subscriber(topics.gripper_feedback, JointState, feedback_cb, queue_size=1)
    rospy.Subscriber(topics.relay_status, String, relay.callback, queue_size=1)
    info(
        "Read-only gripper debug: "
        f"target={topics.gripper_target}, feedback={topics.gripper_feedback}"
    )

    rate = rospy.Rate(args.rate_hz)
    deadline = time.monotonic() + config.system.doctor.ros_topic_timeout_s
    max_age_s = config.system.joint_safety.max_feedback_age_s
    while not rospy.is_shutdown():
        target_mm = target.get(max_age_s=max_age_s)
        feedback_mm = feedback.get(max_age_s=max_age_s)
        if target_mm is not None and feedback_mm is not None:
            try:
                reading = build_gripper_debug_reading(
                    target_mm=target_mm,
                    feedback_mm=feedback_mm,
                    stroke_min_mm=config.system.gripper.stroke_min_mm,
                    stroke_max_mm=config.system.gripper.stroke_max_mm,
                    source_min=config.gripper.source_min,
                    source_max=config.gripper.source_max,
                    invert=config.gripper.invert,
                )
            except ValueError as exc:
                failure(str(exc))
                return 1
            info(
                format_gripper_debug_reading(
                    reading,
                    relay_summary=relay.summary(),
                )
            )
            if args.once:
                return 0
        elif args.once and time.monotonic() >= deadline:
            failure(
                "No fresh gripper target/feedback pair; start Teleop services and bridge first."
            )
            return 1
        rate.sleep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
