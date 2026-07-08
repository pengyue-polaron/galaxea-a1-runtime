#!/usr/bin/env python3
# ruff: noqa: E402
"""SO leader to Galaxea A1 staged joint teleoperation bridge."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_A1_SDK_RUNTIME = ROOT_DIR / "third_party" / "A1_SDK_runtime" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for candidate in (
    str(ROOT_DIR / "third_party" / "lerobot" / "src"),
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK_RUNTIME / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.eef_bridge import (
    RelayStatus,
    decode_relay_status,
    relay_state_summary,
    relay_status_is_fresh,
)
from galaxea_a1_runtime.teleop import (
    JointMappingConfig,
    detect_leader_joint_keys,
    map_leader_joints_to_a1,
    parse_csv_floats,
    parse_csv_strings,
)
from galaxea_a1_runtime.teleop.a1_so_leader import A1SOLeader, SOLeaderTeleopConfig


class LatestCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._value: Any | None = None
        self._updated_monotonic: float | None = None

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._updated_monotonic = time.monotonic()

    def get(self) -> tuple[Any | None, float | None]:
        with self._lock:
            return self._value, self._updated_monotonic


class A1JointStateCache:
    def __init__(self, ordered_names: tuple[str, ...]):
        self.ordered_names = ordered_names
        self.cache = LatestCache()

    def callback(self, msg: JointState) -> None:
        self.cache.set(msg)

    def positions(self) -> tuple[float, ...] | None:
        msg, _ = self.cache.get()
        if msg is None:
            return None
        names = list(getattr(msg, "name", []))
        values = list(getattr(msg, "position", []))
        if len(values) < len(self.ordered_names):
            return None
        name_to_idx = {name: index for index, name in enumerate(names)}
        if names and all(name in name_to_idx for name in self.ordered_names):
            indices = [name_to_idx[name] for name in self.ordered_names]
            if all(index < len(values) for index in indices):
                return tuple(float(values[index]) for index in indices)
        return tuple(float(value) for value in values[: len(self.ordered_names)])


class RelayMonitor:
    def __init__(self, max_status_age_s: float):
        self.max_status_age_s = max_status_age_s
        self.cache = LatestCache()

    def callback(self, msg: String) -> None:
        self.cache.set(decode_relay_status(msg.data))

    def status(self) -> tuple[RelayStatus | None, float | None]:
        value, updated = self.cache.get()
        return value, updated

    def summary(self) -> str:
        status, updated = self.status()
        return relay_state_summary(status, updated, max_age_s=self.max_status_age_s)

    def is_active(self) -> bool:
        status, updated = self.status()
        return (
            relay_status_is_fresh(updated, max_age_s=self.max_status_age_s)
            and (status or RelayStatus("UNKNOWN")).state == "ACTIVE"
        )


def main() -> int:
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("--hz must be positive")

    dof = args.dof
    target_names = parse_csv_strings(args.target_joint_names, dof, "--target-joint-names")
    mapping = JointMappingConfig(
        relative=args.relative,
        input_degrees=args.input_degrees,
        scale=parse_csv_floats(args.scale, dof, "--scale"),
        sign=parse_csv_floats(args.sign, dof, "--sign"),
        bias_rad=parse_csv_floats(args.bias_rad, dof, "--bias-rad"),
        lower_limits=parse_csv_floats(args.lower_limits, dof, "--lower-limits"),
        upper_limits=parse_csv_floats(args.upper_limits, dof, "--upper-limits"),
    )
    mapping.validate(dof)

    rospy.init_node("a1_so100_joint_bridge", anonymous=False, disable_signals=True)
    a1_state = A1JointStateCache(target_names)
    relay = RelayMonitor(args.max_relay_status_age)
    rospy.Subscriber(args.joint_states_topic, JointState, a1_state.callback, queue_size=10)
    rospy.Subscriber(args.relay_status_topic, String, relay.callback, queue_size=10)
    target_pub = rospy.Publisher(args.target_topic, JointState, queue_size=10)
    motion_enable_pub = rospy.Publisher(args.motion_enable_topic, Bool, queue_size=1, latch=True)
    gripper_pub = rospy.Publisher(args.gripper_topic, gripper_position_control, queue_size=10)

    leader = A1SOLeader(
        SOLeaderTeleopConfig(
            id=args.leader_id,
            port=args.leader_port,
            use_degrees=args.leader_use_degrees,
        )
    )
    leader.connect(calibrate=False)

    try:
        rate = rospy.Rate(args.hz)
        print("[teleop bridge] waiting for leader and A1 joint feedback ...", flush=True)
        leader_action0 = leader.get_action()
        leader_keys = detect_leader_joint_keys(leader_action0, dof)
        if args.gripper_enabled and args.gripper_source_key not in leader_action0:
            raise RuntimeError(
                f"gripper key {args.gripper_source_key!r} not in leader action keys: "
                f"{sorted(leader_action0)}"
            )
        leader_start = tuple(float(leader_action0[key]) for key in leader_keys)
        a1_start = wait_for_a1_start(a1_state, timeout_s=args.a1_state_timeout)
        print(f"[teleop bridge] leader_keys={list(leader_keys)}")
        print(f"[teleop bridge] target_names={list(target_names)}")
        print("[teleop bridge] publishing first target while relay is locked")

        first_target = map_leader_joints_to_a1(
            leader_now=leader_start,
            leader_start=leader_start,
            a1_start=a1_start,
            config=mapping,
        )
        publish_target(target_pub, target_names, first_target)
        arm_relay(motion_enable_pub, relay, timeout_s=args.relay_enable_timeout)
        print("[teleop bridge] relay ACTIVE; teleop is live")

        while not rospy.is_shutdown():
            if not relay.is_active():
                motion_enable_pub.publish(Bool(data=False))
                raise RuntimeError(
                    "A1 relay is not confirmed ACTIVE; stopping teleop. "
                    f"Last relay state: {relay.summary()}"
                )
            action = leader.get_action()
            leader_now = tuple(float(action[key]) for key in leader_keys)
            target = map_leader_joints_to_a1(
                leader_now=leader_now,
                leader_start=leader_start,
                a1_start=a1_start,
                config=mapping,
            )
            stamp = publish_target(target_pub, target_names, target)
            if args.gripper_enabled:
                publish_gripper(gripper_pub, action, args, stamp)
            rate.sleep()
    finally:
        motion_enable_pub.publish(Bool(data=False))
        leader.disconnect()
    return 0


def wait_for_a1_start(cache: A1JointStateCache, *, timeout_s: float) -> tuple[float, ...]:
    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        positions = cache.positions()
        if positions is not None:
            return positions
        time.sleep(0.05)
    raise RuntimeError(f"No usable /joint_states_host within {timeout_s:.1f}s")


def publish_target(pub: Any, names: tuple[str, ...], target: tuple[float, ...]) -> Any:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(names)
    msg.position = list(target)
    pub.publish(msg)
    return msg.header.stamp


def arm_relay(pub: Any, relay: RelayMonitor, *, timeout_s: float) -> None:
    pub.publish(Bool(data=True))
    deadline = time.monotonic() + timeout_s
    last = relay.summary()
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        last = relay.summary()
        if relay.is_active():
            return
        status, _ = relay.status()
        if status is not None and status.state == "FAULT":
            break
        time.sleep(0.05)
    pub.publish(Bool(data=False))
    raise RuntimeError(f"A1 relay did not become ACTIVE: {last}")


def publish_gripper(pub: Any, leader_action: dict[str, float], args: argparse.Namespace, stamp: Any) -> None:
    pct = max(0.0, min(100.0, float(leader_action[args.gripper_source_key])))
    if args.gripper_invert:
        pct = 100.0 - pct
    stroke = args.gripper_min_stroke_mm + (
        args.gripper_max_stroke_mm - args.gripper_min_stroke_mm
    ) * (pct / 100.0)
    msg = gripper_position_control()
    msg.header.stamp = stamp
    msg.gripper_stroke = float(stroke)
    pub.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO leader -> staged A1 joint teleoperation bridge")
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--leader-use-degrees", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--dof", type=int, default=6)
    parser.add_argument("--joint-states-topic", default="/joint_states_host")
    parser.add_argument("--target-topic", default="/arm_joint_target_position")
    parser.add_argument(
        "--target-joint-names",
        default="arm_joint1,arm_joint2,arm_joint3,arm_joint4,arm_joint5,arm_joint6",
    )
    parser.add_argument("--relative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-degrees", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale", default="1,1,1,1,1,1")
    parser.add_argument("--sign", default="-1,1,1,-1,1,-1")
    parser.add_argument("--bias-rad", default="0,0,0,0,0,0")
    parser.add_argument("--lower-limits", default="-2.8798,0,-3.3161,-2.8798,-1.6581,-2.8798")
    parser.add_argument("--upper-limits", default="2.8798,3.2289,0,2.8798,1.6581,2.8798")
    parser.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--relay-enable-timeout", type=float, default=2.0)
    parser.add_argument("--max-relay-status-age", type=float, default=1.0)
    parser.add_argument("--a1-state-timeout", type=float, default=10.0)
    parser.add_argument("--gripper-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-source-key", default="gripper.pos")
    parser.add_argument(
        "--gripper-topic",
        "--gripper-position-topic",
        dest="gripper_topic",
        default="/gripper_position_control_host",
    )
    parser.add_argument("--gripper-min-stroke-mm", type=float, default=0.0)
    parser.add_argument("--gripper-max-stroke-mm", type=float, default=200.0)
    parser.add_argument("--gripper-invert", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
