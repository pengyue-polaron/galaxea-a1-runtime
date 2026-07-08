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
from signal_arm.msg import arm_control, status_stamped
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.constants import (  # noqa: E402
    DEFAULT_MAX_COMMAND_AGE_S,
    DEFAULT_MAX_INITIAL_COMMAND_ERROR_RAD,
    DEFAULT_RELAY_ARMING_TIMEOUT_S,
)
from galaxea_a1_runtime.safety import (  # noqa: E402
    RelayInputs,
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
        self.motor_errors = ()
        self.status_time = 0.0
        self.requested_enabled = False
        self.fault_reason = ""
        self.initial_alignment_checked = False

        self.command_pub = rospy.Publisher(args.output_topic, arm_control, queue_size=1)
        self.status_pub = rospy.Publisher(args.relay_status_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(args.joint_topic, JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber(args.input_topic, arm_control, self._command_cb, queue_size=1)
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

    def _snapshot(self):
        now = time.monotonic()
        with self.lock:
            joints = list(self.joints)
            command = None if self.command is None else copy.deepcopy(self.command)
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
        return now, joints, command, inputs, fault_reason

    def _publish_status(self, state, reason, inputs):
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
            now, joints, command, inputs, fault_reason = self._snapshot()
            reason = relay_block_reason(inputs, arm_joints=self.args.arm_joints, max_age=self.args.max_input_age)

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

                command.header.stamp = rospy.Time.now()
                p_des = list(command.p_des)
                p_des[: self.args.arm_joints] = output
                command.p_des = p_des
                self.command_pub.publish(command)

            if inputs.enabled and state == "ARMING" and reason:
                oldest = max(inputs.joint_age, inputs.source_age, inputs.status_age)
                if oldest > self.args.arming_timeout:
                    self._latch_fault(reason)

            if now - last_status_time >= status_period:
                self._publish_status(state, reason or "", inputs)
                last_status_time = now
            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/arm_joint_command_a1_staged")
    parser.add_argument("--output-topic", default="/arm_joint_command_host")
    parser.add_argument("--joint-topic", default="/joint_states_host")
    parser.add_argument("--motor-status-topic", default="/arm_status_host")
    parser.add_argument("--enable-topic", default="/a1_arm_motion_enable")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--arm-joints", type=int, default=6)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--status-rate", type=float, default=5.0)
    parser.add_argument("--max-input-age", type=float, default=DEFAULT_MAX_COMMAND_AGE_S)
    parser.add_argument("--arming-timeout", type=float, default=DEFAULT_RELAY_ARMING_TIMEOUT_S)
    parser.add_argument("--max-initial-error", type=float, default=DEFAULT_MAX_INITIAL_COMMAND_ERROR_RAD)
    return parser.parse_args()


def main():
    rospy.init_node("safe_arm_command_relay", anonymous=False)
    SafeArmCommandRelay(parse_args()).run()


if __name__ == "__main__":
    main()
