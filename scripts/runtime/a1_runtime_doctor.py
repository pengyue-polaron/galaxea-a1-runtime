#!/usr/bin/env python3
"""Layered health check for the Galaxea A1 execution runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from galaxea_a1_runtime.configuration.system import (
    SYSTEM_CONFIG,
    load_system_config,
)
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.constants import EE_TRACKER_NODE, SAFE_RELAY_NODE
from galaxea_a1_runtime.runtime.health_checks import (
    Check,
    RosDoctorContext,
    add_check,
    arm_control_result,
    finish_checks,
    relay_status_result,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument("--system-config", type=Path, default=ROOT / SYSTEM_CONFIG)
    parser.add_argument("--tracker-node", default=EE_TRACKER_NODE)
    args = parser.parse_args()
    system = load_system_config(args.system_config, repo_root=ROOT)

    checks: list[Check] = []
    serial = Path(system.host.a1_serial)
    serial_ok = serial.exists()
    serial_detail = "missing (expected while arm power is off)"
    if serial_ok:
        serial_detail = f"{serial} -> {serial.resolve()}"
    add_check(
        checks, "serial", serial_ok, serial_detail, required=args.require_execution
    )

    try:
        import rospy
        import rosgraph
        import rosnode
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from signal_arm.msg import arm_control, status_stamped
        from std_msgs.msg import String
    except Exception as exc:
        add_check(
            checks, "ros_python", False, repr(exc), required=args.require_execution
        )
        return finish_checks(checks, json_output=args.json)

    master_ok = rosgraph.is_master_online()
    add_check(
        checks,
        "ros_master",
        master_ok,
        os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
        required=args.require_execution,
    )
    if not master_ok:
        return finish_checks(checks, json_output=args.json)

    rospy.init_node("a1_runtime_doctor", anonymous=True, disable_signals=True)
    ros = RosDoctorContext(
        rospy=rospy,
        rosnode=rosnode,
        checks=checks,
        timeout_s=system.doctor.ros_topic_timeout_s,
        required=args.require_execution,
    )
    ros.message(
        "joint_feedback",
        system.topics.joint_states,
        JointState,
        lambda msg: (
            len(msg.position) >= len(system.joint_safety.names),
            f"positions={len(msg.position)} "
            f"arm_values={list(msg.position[: len(system.joint_safety.names)])}",
        ),
    )
    ros.motor_status(system.topics.motor_status, status_stamped)
    ros.message(
        "ee_feedback",
        system.topics.eef_pose,
        PoseStamped,
        lambda msg: (
            True,
            "xyz="
            + str(
                [
                    round(msg.pose.position.x, 4),
                    round(msg.pose.position.y, 4),
                    round(msg.pose.position.z, 4),
                ]
            ),
        ),
    )
    ros.message(
        "staged_command",
        system.topics.staged_command,
        arm_control,
        lambda msg: arm_control_result(
            msg,
            arm_joints=len(system.joint_safety.names),
            allowed_modes=system.relay.allowed_control_modes,
        ),
    )
    ros.node("tracker", args.tracker_node)
    ros.node("relay", SAFE_RELAY_NODE)
    try:
        publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        publisher_map = {topic: set(names) for topic, names in publishers}
        gripper_publishers = publisher_map.get(system.topics.gripper_command, set())
        add_check(
            checks,
            "gripper_relay_ownership",
            gripper_publishers == {SAFE_RELAY_NODE},
            f"{system.topics.gripper_command} publishers={sorted(gripper_publishers)}",
            required=args.require_execution,
        )
    except Exception as exc:
        add_check(
            checks,
            "gripper_relay_ownership",
            False,
            repr(exc),
            required=args.require_execution,
        )
    ros.message(
        "relay_state",
        system.topics.relay_status,
        String,
        lambda msg: relay_status_result(msg, require_execution=args.require_execution),
    )

    return finish_checks(checks, json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
