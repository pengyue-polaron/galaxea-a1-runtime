#!/usr/bin/env python3
"""Layered health check for the Galaxea A1 execution runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    level: str
    detail: str


def add(checks, name, ok, detail, *, required=True):
    checks.append(Check(name, "PASS" if ok else ("FAIL" if required else "WARN"), detail))


def add_level(checks, name, level, detail):
    checks.append(Check(name, level, detail))


def finish(checks, json_output):
    if json_output:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        width = max((len(item.name) for item in checks), default=0)
        for item in checks:
            print(f"[{item.level:4}] {item.name:<{width}}  {item.detail}")
    return 1 if any(item.level == "FAIL" for item in checks) else 0


def relay_status(msg, require_execution):
    try:
        payload = json.loads(msg.data)
    except Exception:
        return False, f"invalid JSON: {msg.data!r}"
    state = payload.get("state", "UNKNOWN")
    ok_states = {"LOCKED", "ACTIVE"} if require_execution else {"LOCKED", "ARMING", "FAULT", "ACTIVE"}
    return state in ok_states, f"{state}: {payload.get('reason', '')}"


IDLE_TIMEOUT_CODE = 1 << 6


def motor_status_level(msg):
    names = list(msg.data.name)
    errors = list(msg.data.motor_errors)
    rows = []
    bad = []
    idle_timeout = []
    if len(errors) < 7:
        return "FAIL", f"motor status has {len(errors)} entries, need 7"
    for idx, item in enumerate(errors):
        name = names[idx] if idx < len(names) else f"motor{idx + 1}"
        code = int(item.error_code)
        desc = "; ".join(str(part) for part in item.error_description) or "OK"
        rows.append(f"{idx + 1}:{name}=code{code}({desc})")
        if idx < 7 and code == IDLE_TIMEOUT_CODE:
            idle_timeout.append(name)
        elif idx < 7 and code != 0:
            bad.append((name, code))
    detail = "; ".join(rows)
    if bad:
        return "FAIL", detail
    if idle_timeout:
        return "WARN", "idle ECU->ACU timeout treated as non-blocking; " + detail
    return "PASS", detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument("--serial", default="/dev/a1")
    parser.add_argument("--staged-command-topic", default="/arm_joint_command_a1_staged")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--tracker-node", default="/eeTracker_demo_node")
    args = parser.parse_args()

    checks = []
    serial = Path(args.serial)
    serial_ok = serial.exists()
    serial_detail = "missing (expected while arm power is off)"
    if serial_ok:
        serial_detail = f"{serial} -> {serial.resolve()}"
    add(checks, "serial", serial_ok, serial_detail, required=args.require_execution)

    try:
        import rospy
        import rosgraph
        import rosnode
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from signal_arm.msg import arm_control, status_stamped
        from std_msgs.msg import String
    except Exception as exc:
        add(checks, "ros_python", False, repr(exc), required=args.require_execution)
        return finish(checks, args.json)

    master_ok = rosgraph.is_master_online()
    add(
        checks,
        "ros_master",
        master_ok,
        os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
        required=args.require_execution,
    )
    if not master_ok:
        return finish(checks, args.json)

    rospy.init_node("a1_runtime_doctor", anonymous=True, disable_signals=True)
    topics = dict(rospy.get_published_topics())

    def message_check(name, topic, cls, validator, required):
        if topic not in topics:
            add(checks, name, False, f"{topic} not published", required=required)
            return None
        try:
            msg = rospy.wait_for_message(topic, cls, timeout=args.timeout)
            ok, detail = validator(msg)
            add(checks, name, ok, detail, required=required)
            return msg
        except Exception as exc:
            add(checks, name, False, f"{topic}: {exc}", required=required)
            return None

    def motor_status_check(required):
        topic = "/arm_status_host"
        if topic not in topics:
            add(checks, "motor_status", False, f"{topic} not published", required=required)
            return None
        try:
            msg = rospy.wait_for_message(topic, status_stamped, timeout=args.timeout)
            level, detail = motor_status_level(msg)
            if level == "FAIL" and not required:
                level = "WARN"
            add_level(checks, "motor_status", level, detail)
            return msg
        except Exception as exc:
            add(checks, "motor_status", False, f"{topic}: {exc}", required=required)
            return None

    message_check(
        "joint_feedback",
        "/joint_states_host",
        JointState,
        lambda msg: (
            len(msg.position) >= 7,
            f"positions={len(msg.position)} values={list(msg.position[:7])}",
        ),
        args.require_execution,
    )
    motor_status_check(args.require_execution)
    message_check(
        "ee_feedback",
        "/end_effector_pose",
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
        args.require_execution,
    )
    message_check(
        "staged_command",
        args.staged_command_topic,
        arm_control,
        lambda msg: (len(msg.p_des) >= 6, f"p_des_count={len(msg.p_des)}"),
        args.require_execution,
    )

    nodes = set(rosnode.get_node_names())

    def node_alive(name):
        if name not in nodes:
            return False
        try:
            return bool(rosnode.rosnode_ping(name, max_count=1, verbose=False))
        except Exception:
            return False

    tracker_alive = node_alive(args.tracker_node)
    add(
        checks,
        "tracker",
        tracker_alive,
        f"{args.tracker_node} responds to XML-RPC" if tracker_alive else f"{args.tracker_node} missing or stale",
        required=args.require_execution,
    )
    relay_alive = node_alive("/safe_arm_command_relay")
    add(
        checks,
        "relay",
        relay_alive,
        "relay responds to XML-RPC" if relay_alive else "missing or stale registration",
        required=args.require_execution,
    )
    message_check(
        "relay_state",
        args.relay_status_topic,
        String,
        lambda msg: relay_status(msg, args.require_execution),
        args.require_execution,
    )

    return finish(checks, args.json)


if __name__ == "__main__":
    sys.exit(main())
