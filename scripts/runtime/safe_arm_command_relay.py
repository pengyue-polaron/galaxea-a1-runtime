#!/usr/bin/env python3
# ruff: noqa: E402
"""Fail-closed command relay for the Galaxea A1 arm."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control, status_stamped
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.safety import (  # noqa: E402
    RelayInputs,
    actuator_error_block_reason,
    gripper_stroke_block_reason,
    relay_block_reason,
    validate_initial_alignment,
)


class SafeArmCommandRelay:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.joints = []
        self.joint_time = 0.0
        self.command = None
        self.command_time = 0.0
        self.gripper_command = None
        self.gripper_command_time = 0.0
        self.motor_errors = ()
        self.status_time = 0.0
        self.requested_enabled = False
        self.fault_reason = ""
        self.initial_alignment_checked = False

        self.command_pub = rospy.Publisher(args.output_topic, arm_control, queue_size=1)
        self.gripper_pub = rospy.Publisher(
            args.gripper_output_topic,
            gripper_position_control,
            queue_size=1,
        )
        self.status_pub = rospy.Publisher(args.relay_status_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(args.joint_topic, JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber(args.input_topic, arm_control, self._command_cb, queue_size=1)
        rospy.Subscriber(
            args.gripper_input_topic,
            gripper_position_control,
            self._gripper_command_cb,
            queue_size=1,
        )
        rospy.Subscriber(args.motor_status_topic, status_stamped, self._status_cb, queue_size=1)
        rospy.Subscriber(args.enable_topic, Bool, self._enable_cb, queue_size=1)

    def _enable_cb(self, msg):
        with self.lock:
            requested = bool(msg.data)
            if requested and not self.requested_enabled:
                self.fault_reason = ""
                self.initial_alignment_checked = False
            self.requested_enabled = requested

    def _joint_cb(self, msg):
        values = [float(v) for v in msg.position[: self.args.arm_joints]]
        if values and all(math.isfinite(v) for v in values):
            with self.lock:
                self.joints = values
                self.joint_time = time.monotonic()

    def _command_cb(self, msg):
        values = [float(v) for v in msg.p_des[: self.args.arm_joints]]
        if values and all(math.isfinite(v) for v in values):
            with self.lock:
                self.command = copy.deepcopy(msg)
                self.command_time = time.monotonic()

    def _status_cb(self, msg):
        with self.lock:
            self.motor_errors = tuple(int(item.error_code) for item in msg.data.motor_errors)
            self.status_time = time.monotonic()

    def _gripper_command_cb(self, msg):
        stroke = float(msg.gripper_stroke)
        reason = gripper_stroke_block_reason(
            stroke,
            minimum_mm=self.args.gripper_min_stroke_mm,
            maximum_mm=self.args.gripper_max_stroke_mm,
        )
        if reason is not None:
            self._latch_fault(reason)
            return
        with self.lock:
            self.gripper_command = copy.deepcopy(msg)
            self.gripper_command_time = time.monotonic()

    def _snapshot(self):
        now = time.monotonic()
        with self.lock:
            joints = list(self.joints)
            command = None if self.command is None else copy.deepcopy(self.command)
            gripper_command = (
                None if self.gripper_command is None else copy.deepcopy(self.gripper_command)
            )
            gripper_age = now - self.gripper_command_time
            inputs = RelayInputs(
                enabled=self.requested_enabled,
                joint_age=now - self.joint_time,
                source_age=now - self.command_time,
                status_age=now - self.status_time,
                joint_count=len(self.joints),
                source_count=0 if self.command is None else len(self.command.p_des),
                motor_error_codes=self.motor_errors,
            )
            fault_reason = self.fault_reason
        return now, joints, command, gripper_command, gripper_age, inputs, fault_reason

    def _publish_status(self, state, reason, inputs, gripper_age, gripper_forwarding):
        payload = {
            "state": state,
            "reason": reason,
            "enabled_requested": inputs.enabled,
            "joint_age_s": round(inputs.joint_age, 3),
            "source_age_s": round(inputs.source_age, 3),
            "status_age_s": round(inputs.status_age, 3),
            "joint_count": inputs.joint_count,
            "source_count": inputs.source_count,
            "motor_error_codes": list(inputs.motor_error_codes),
            "gripper_age_s": None if gripper_age == float("inf") else round(gripper_age, 3),
            "gripper_forwarding": gripper_forwarding,
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _latch_fault(self, reason):
        with self.lock:
            if not self.fault_reason:
                self.fault_reason = reason
                self.requested_enabled = False
        rospy.logerr("A1 relay FAULT: %s", reason)

    def run(self):
        rate = rospy.Rate(self.args.rate)
        status_period = 1.0 / self.args.status_rate
        last_status_time = 0.0
        while not rospy.is_shutdown():
            now, joints, command, gripper_command, gripper_age, inputs, fault_reason = self._snapshot()
            reason = relay_block_reason(inputs, arm_joints=self.args.arm_joints, max_age=self.args.max_input_age)
            gripper_forwarding = False

            if fault_reason:
                state, reason = "FAULT", fault_reason
            elif reason == "locked":
                state = "LOCKED"
            elif reason is not None:
                state = "ARMING"
            else:
                state = "ACTIVE"
                raw = [float(v) for v in command.p_des[: self.args.arm_joints]]
                try:
                    if not self.initial_alignment_checked:
                        validate_initial_alignment(
                            joints,
                            raw,
                            max_abs_error=self.args.max_initial_error,
                        )
                        self.initial_alignment_checked = True
                        rospy.logwarn("A1 relay ACTIVE; pass-through joint commands enabled")
                    output = list(raw)
                except ValueError as exc:
                    self._latch_fault(str(exc))
                    rate.sleep()
                    continue

                gripper_ready = gripper_command is not None and gripper_age <= self.args.max_input_age
                if gripper_ready:
                    gripper_reason = actuator_error_block_reason(
                        inputs.motor_error_codes,
                        index=self.args.arm_joints,
                        label="gripper",
                    )
                    if gripper_reason is not None:
                        self._latch_fault(gripper_reason)
                        rate.sleep()
                        continue

                command.header.stamp = rospy.Time.now()
                p_des = list(command.p_des)
                p_des[: self.args.arm_joints] = output
                command.p_des = p_des
                self.command_pub.publish(command)
                if gripper_ready:
                    gripper_command.header.stamp = rospy.Time.now()
                    self.gripper_pub.publish(gripper_command)
                    gripper_forwarding = True

            if inputs.enabled and state == "ARMING" and reason:
                oldest = max(inputs.joint_age, inputs.source_age, inputs.status_age)
                if oldest > self.args.arming_timeout:
                    self._latch_fault(reason)

            if now - last_status_time >= status_period:
                self._publish_status(
                    state,
                    reason or "",
                    inputs,
                    gripper_age if gripper_command is not None else float("inf"),
                    gripper_forwarding,
                )
                last_status_time = now
            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--joint-topic", required=True)
    parser.add_argument("--motor-status-topic", required=True)
    parser.add_argument("--enable-topic", required=True)
    parser.add_argument("--relay-status-topic", required=True)
    parser.add_argument("--gripper-input-topic", required=True)
    parser.add_argument("--gripper-output-topic", required=True)
    parser.add_argument("--gripper-min-stroke-mm", type=float, required=True)
    parser.add_argument("--gripper-max-stroke-mm", type=float, required=True)
    parser.add_argument("--arm-joints", type=int, default=6)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--status-rate", type=float, default=5.0)
    parser.add_argument("--max-input-age", type=float, required=True)
    parser.add_argument("--arming-timeout", type=float, required=True)
    parser.add_argument("--max-initial-error", type=float, required=True)
    args = parser.parse_args()
    if min(args.rate, args.status_rate, args.max_input_age, args.arming_timeout) <= 0:
        parser.error("relay rates and timeouts must be positive")
    if args.max_initial_error < 0:
        parser.error("--max-initial-error must be non-negative")
    if args.gripper_min_stroke_mm >= args.gripper_max_stroke_mm:
        parser.error("gripper minimum must be below maximum")
    return args


def main():
    rospy.init_node("safe_arm_command_relay", anonymous=False)
    SafeArmCommandRelay(parse_args()).run()


if __name__ == "__main__":
    main()
