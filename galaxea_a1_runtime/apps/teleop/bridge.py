#!/usr/bin/env python3
# ruff: noqa: E402
"""SO leader to Galaxea A1 staged joint bridge implementation."""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR)

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.gripper import denormalize_stroke, normalize_source_position
from galaxea_a1_runtime.console import info
from galaxea_a1_runtime.runtime.relay import RelayMonitor
from galaxea_a1_runtime.runtime.ros_feedback import (
    A1JointStateCache,
    StagedCommandMonitor,
)
from galaxea_a1_runtime.teleop import detect_leader_joint_keys, map_leader_joints_to_a1
from lerobot_teleoperator_galaxea_a1_so_leader import (
    GalaxeaA1SOLeader,
    GalaxeaA1SOLeaderConfig,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def log(message: str) -> None:
    info(message.removeprefix("[teleop bridge] "))


def run(config: TeleopConfig) -> int:
    system = config.system
    topics = system.topics
    bridge = config.bridge
    dof = bridge.dof
    target_names = system.joint_safety.names
    mapping = bridge.mapping

    rospy.init_node("a1_so100_joint_bridge", anonymous=False, disable_signals=True)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    a1_state = A1JointStateCache(target_names)
    relay = RelayMonitor(system.relay.max_status_age_s)
    staged = StagedCommandMonitor()
    rospy.Subscriber(topics.joint_states, JointState, a1_state.callback, queue_size=10)
    rospy.Subscriber(topics.relay_status, String, relay.callback, queue_size=10)
    rospy.Subscriber(topics.staged_command, arm_control, staged.callback, queue_size=10)
    target_pub = rospy.Publisher(topics.joint_target, JointState, queue_size=10)
    motion_enable_pub = rospy.Publisher(
        topics.motion_enable, Bool, queue_size=1, latch=True
    )
    gripper_pub = rospy.Publisher(
        topics.gripper_target, gripper_position_control, queue_size=10
    )

    leader = GalaxeaA1SOLeader(
        GalaxeaA1SOLeaderConfig(
            id=config.leader.id,
            port=config.leader.port,
        )
    )
    leader.connect(calibrate=False)

    try:
        rate = rospy.Rate(bridge.hz)
        log("[teleop bridge] waiting for leader and A1 joint feedback ...")
        leader_action0 = leader.get_action()
        leader_keys = detect_leader_joint_keys(leader_action0, dof)
        if config.gripper.enabled and config.gripper.source_key not in leader_action0:
            raise RuntimeError(
                f"gripper key {config.gripper.source_key!r} not in leader action keys: "
                f"{sorted(leader_action0)}"
            )
        leader_start = tuple(float(leader_action0[key]) for key in leader_keys)
        log(f"[teleop bridge] leader_keys={list(leader_keys)}")
        a1_start = wait_for_a1_start(
            a1_state,
            timeout_s=bridge.a1_state_timeout_s,
            topic=topics.joint_states,
        )
        log(f"[teleop bridge] target_names={list(target_names)}")
        log(f"[teleop bridge] a1_start={[round(value, 4) for value in a1_start]}")
        log("[teleop bridge] publishing first target while relay is locked")

        first_target = map_leader_joints_to_a1(
            leader_now=leader_start,
            leader_start=leader_start,
            a1_start=a1_start,
            config=mapping,
        )
        wait_for_staged_alignment(
            target_pub,
            target_names,
            first_target,
            staged,
            dof=dof,
            timeout_s=bridge.a1_state_timeout_s,
            hz=bridge.hz,
            tolerance_rad=system.joint_safety.initial_alignment_tolerance_rad,
        )
        log("[teleop bridge] tracker staged output aligned with first target")
        arm_relay(
            motion_enable_pub,
            relay,
            timeout_s=system.relay.enable_timeout_s,
        )
        log("[teleop bridge] relay ACTIVE; teleop is live")

        last_loop_log = time.monotonic()
        loop_count = 0
        while not rospy.is_shutdown() and not stop_requested.is_set():
            if not relay.is_active():
                if stop_requested.is_set():
                    break
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
            if config.gripper.enabled:
                publish_gripper(gripper_pub, action, config, stamp)
            loop_count += 1
            now = time.monotonic()
            if now - last_loop_log >= 2.0:
                log(
                    f"[teleop bridge] publishing target at {loop_count / (now - last_loop_log):.1f} Hz"
                )
                loop_count = 0
                last_loop_log = now
            rate.sleep()
    finally:
        motion_enable_pub.publish(Bool(data=False))
        time.sleep(0.1)
        leader.disconnect()
        log("[teleop bridge] stopped; relay disabled")
    return 0


def wait_for_a1_start(
    cache: A1JointStateCache, *, timeout_s: float, topic: str
) -> tuple[float, ...]:
    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        positions = cache.positions()
        if positions is not None:
            return positions
        time.sleep(0.05)
    raise RuntimeError(f"No usable {topic} within {timeout_s:.1f}s")


def publish_target(pub: Any, names: tuple[str, ...], target: tuple[float, ...]) -> Any:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(names)
    msg.position = list(target)
    pub.publish(msg)
    return msg.header.stamp


def wait_for_staged_alignment(
    pub: Any,
    names: tuple[str, ...],
    target: tuple[float, ...],
    staged: StagedCommandMonitor,
    *,
    dof: int,
    timeout_s: float,
    hz: float,
    tolerance_rad: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    period = 1.0 / hz
    last_error: float | None = None
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        publish_target(pub, names, target)
        last_error = staged.max_error(target, dof)
        if last_error is not None and last_error <= tolerance_rad:
            return
        time.sleep(period)
    detail = (
        "no staged command"
        if last_error is None
        else f"last max error {last_error:.4f} rad"
    )
    raise RuntimeError(
        "Tracker staged output did not align with the initial target within "
        f"{timeout_s:.1f}s ({detail}, tolerance {tolerance_rad:.4f} rad)"
    )


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


def publish_gripper(
    pub: Any, leader_action: dict[str, float], config: TeleopConfig, stamp: Any
) -> None:
    normalized = normalize_source_position(
        leader_action[config.gripper.source_key],
        source_min=config.gripper.source_min,
        source_max=config.gripper.source_max,
        invert=config.gripper.invert,
        saturate_out_of_range=config.gripper.saturate_out_of_range,
    )
    stroke = denormalize_stroke(
        normalized,
        stroke_min_mm=config.system.gripper.stroke_min_mm,
        stroke_max_mm=config.system.gripper.stroke_max_mm,
    )
    msg = gripper_position_control()
    msg.header.stamp = stamp
    msg.gripper_stroke = float(stroke)
    pub.publish(msg)
