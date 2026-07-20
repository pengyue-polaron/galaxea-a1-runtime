#!/usr/bin/env python3.12
# ruff: noqa: E402
"""Fail-closed command relay for the Galaxea A1 arm."""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python  # noqa: E402

configure_ros1_python(ROOT)

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control, status_stamped
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.safety import (  # noqa: E402
    RelayInputs,
    actuator_error_block_reason,
    gripper_stroke_block_reason,
    relay_block_reason,
    require_finite_vector,
    validate_arm_control_command,
    validate_initial_alignment,
)
from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from galaxea_a1_runtime.constants import SAFE_RELAY_NODE_NAME  # noqa: E402
from galaxea_a1_runtime.configuration.system import (  # noqa: E402
    SYSTEM_CONFIG,
    load_system_config,
)


@dataclass(frozen=True)
class RelayRuntimeConfig:
    input_topic: str
    output_topic: str
    joint_topic: str
    motor_status_topic: str
    enable_topic: str
    relay_status_topic: str
    gripper_input_topic: str
    gripper_output_topic: str
    gripper_min_stroke_mm: float
    gripper_max_stroke_mm: float
    arm_joints: int
    rate: float
    status_rate: float
    max_input_age: float
    max_status_age: float
    arming_timeout: float
    max_initial_error: float
    gripper_ignored_error_mask: int
    allowed_control_modes: tuple[int, ...]


class SafeArmCommandRelay:
    def __init__(self, config: RelayRuntimeConfig):
        self.config = config
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

        self.command_pub = rospy.Publisher(
            config.output_topic, arm_control, queue_size=1
        )
        self.gripper_pub = rospy.Publisher(
            config.gripper_output_topic,
            gripper_position_control,
            queue_size=1,
        )
        self.status_pub = rospy.Publisher(
            config.relay_status_topic, String, queue_size=1, latch=True
        )
        rospy.Subscriber(config.joint_topic, JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber(
            config.input_topic, arm_control, self._command_cb, queue_size=1
        )
        rospy.Subscriber(
            config.gripper_input_topic,
            gripper_position_control,
            self._gripper_command_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            config.motor_status_topic, status_stamped, self._status_cb, queue_size=1
        )
        rospy.Subscriber(config.enable_topic, Bool, self._enable_cb, queue_size=1)

    def _enable_cb(self, msg):
        with self.lock:
            requested = bool(msg.data)
            if requested and not self.requested_enabled:
                self.fault_reason = ""
                self.initial_alignment_checked = False
            self.requested_enabled = requested

    def _joint_cb(self, msg):
        try:
            values = require_finite_vector(
                msg.position,
                count=self.config.arm_joints,
                label="joint feedback",
            )
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            with self.lock:
                self.joints = []
                self.joint_time = 0.0
            self._latch_fault(str(exc))
            return
        with self.lock:
            self.joints = list(values)
            self.joint_time = time.monotonic()

    def _command_cb(self, msg):
        try:
            validate_arm_control_command(
                p_des=msg.p_des,
                v_des=msg.v_des,
                kp=msg.kp,
                kd=msg.kd,
                t_ff=msg.t_ff,
                mode=msg.mode,
                arm_joints=self.config.arm_joints,
                allowed_modes=self.config.allowed_control_modes,
            )
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            with self.lock:
                self.command = None
                self.command_time = 0.0
            self._latch_fault(str(exc))
            return
        with self.lock:
            self.command = copy.deepcopy(msg)
            self.command_time = time.monotonic()

    def _status_cb(self, msg):
        try:
            errors = tuple(int(item.error_code) for item in msg.data.motor_errors)
            if len(errors) < self.config.arm_joints + 1:
                raise ValueError(
                    f"motor status has {len(errors)} entries, "
                    f"need {self.config.arm_joints + 1}"
                )
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            with self.lock:
                self.motor_errors = ()
                self.status_time = 0.0
            self._latch_fault(str(exc))
            return
        with self.lock:
            self.motor_errors = errors
            self.status_time = time.monotonic()

    def _gripper_command_cb(self, msg):
        try:
            stroke = float(msg.gripper_stroke)
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            with self.lock:
                self.gripper_command = None
                self.gripper_command_time = 0.0
            self._latch_fault(f"invalid gripper target: {exc}")
            return
        reason = gripper_stroke_block_reason(
            stroke,
            minimum_mm=self.config.gripper_min_stroke_mm,
            maximum_mm=self.config.gripper_max_stroke_mm,
        )
        if reason is not None:
            with self.lock:
                self.gripper_command = None
                self.gripper_command_time = 0.0
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
                None
                if self.gripper_command is None
                else copy.deepcopy(self.gripper_command)
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
            "gripper_age_s": None
            if gripper_age == float("inf")
            else round(gripper_age, 3),
            "gripper_forwarding": gripper_forwarding,
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _latch_fault(self, reason):
        with self.lock:
            newly_latched = not self.fault_reason
            if not self.fault_reason:
                self.fault_reason = reason
                self.requested_enabled = False
        if newly_latched:
            rospy.logerr("A1 relay FAULT: %s", reason)

    def run(self):
        rate = rospy.Rate(self.config.rate)
        status_period = 1.0 / self.config.status_rate
        last_status_time = 0.0
        while not rospy.is_shutdown():
            now, joints, command, gripper_command, gripper_age, inputs, fault_reason = (
                self._snapshot()
            )
            reason = relay_block_reason(
                inputs,
                arm_joints=self.config.arm_joints,
                max_input_age=self.config.max_input_age,
                max_status_age=self.config.max_status_age,
            )
            gripper_forwarding = False

            if fault_reason:
                state, reason = "FAULT", fault_reason
            elif reason == "locked":
                state = "LOCKED"
            elif reason is not None:
                state = "ARMING"
            else:
                state = "ACTIVE"
                raw = [float(v) for v in command.p_des[: self.config.arm_joints]]
                try:
                    if not self.initial_alignment_checked:
                        validate_initial_alignment(
                            joints,
                            raw,
                            max_abs_error=self.config.max_initial_error,
                        )
                        self.initial_alignment_checked = True
                        rospy.logwarn(
                            "A1 relay ACTIVE; pass-through joint commands enabled"
                        )
                    output = list(raw)
                except ValueError as exc:
                    self._latch_fault(str(exc))
                    rate.sleep()
                    continue

                gripper_ready = (
                    gripper_command is not None
                    and gripper_age <= self.config.max_input_age
                )
                if gripper_ready:
                    gripper_reason = actuator_error_block_reason(
                        inputs.motor_error_codes,
                        index=self.config.arm_joints,
                        label="gripper",
                        ignored_mask=self.config.gripper_ignored_error_mask,
                    )
                    if gripper_reason is not None:
                        self._latch_fault(gripper_reason)
                        rate.sleep()
                        continue

                command.header.stamp = rospy.Time.now()
                p_des = list(command.p_des)
                p_des[: self.config.arm_joints] = output
                command.p_des = p_des
                self.command_pub.publish(command)
                if gripper_ready:
                    gripper_command.header.stamp = rospy.Time.now()
                    self.gripper_pub.publish(gripper_command)
                    gripper_forwarding = True

            if inputs.enabled and state == "ARMING" and reason:
                oldest = max(inputs.joint_age, inputs.source_age, inputs.status_age)
                if oldest > self.config.arming_timeout:
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


def parse_args() -> RelayRuntimeConfig:
    parser = ArgumentParser(
        description="Fail-closed relay configured from the shared system TOML"
    )
    parser.add_argument("--config", type=Path, default=ROOT / SYSTEM_CONFIG)
    cli = parser.parse_args()
    config = load_system_config(cli.config, repo_root=ROOT)
    topics = config.topics
    return RelayRuntimeConfig(
        input_topic=topics.staged_command,
        output_topic=topics.host_command,
        joint_topic=topics.joint_states,
        motor_status_topic=topics.motor_status,
        enable_topic=topics.motion_enable,
        relay_status_topic=topics.relay_status,
        gripper_input_topic=topics.gripper_target,
        gripper_output_topic=topics.gripper_command,
        gripper_min_stroke_mm=config.gripper.stroke_min_mm,
        gripper_max_stroke_mm=config.gripper.stroke_max_mm,
        arm_joints=len(config.joint_safety.names),
        rate=config.relay.rate_hz,
        status_rate=config.relay.status_rate_hz,
        max_input_age=config.relay.max_input_age_s,
        max_status_age=config.relay.max_status_age_s,
        arming_timeout=config.relay.arming_timeout_s,
        max_initial_error=config.joint_safety.initial_alignment_tolerance_rad,
        gripper_ignored_error_mask=config.relay.gripper_ignored_error_mask,
        allowed_control_modes=config.relay.allowed_control_modes,
    )


def main():
    config = parse_args()
    rospy.init_node(SAFE_RELAY_NODE_NAME, anonymous=False)
    SafeArmCommandRelay(config).run()


if __name__ == "__main__":
    main()
