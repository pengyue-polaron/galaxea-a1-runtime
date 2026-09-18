#!/usr/bin/env python3.12
# ruff: noqa: E402
"""Publish validated read-only vendor telemetry for the one-way ROS bridge."""

from __future__ import annotations

import copy
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.configuration.system import (  # noqa: E402
    SYSTEM_CONFIG,
    SystemConfig,
    load_system_config,
)
from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from galaxea_a1_runtime.observability import (  # noqa: E402
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_OK,
    DiagnosticFinding,
    freshness_age_s,
    motor_diagnostic,
    relay_diagnostic,
)
from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python  # noqa: E402
from galaxea_a1_runtime.safety import (  # noqa: E402
    gripper_stroke_block_reason,
    require_finite_vector,
    validate_arm_control_command,
)

configure_ros1_python(ROOT)

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control, status_stamped
from std_msgs.msg import String
from galaxea_a1_runtime.runtime.ros2 import LEGACY_DIAGNOSTICS_TOPIC


class LegacyTelemetryNode:
    def __init__(self, system: SystemConfig) -> None:
        self.system = system
        self._lock = threading.RLock()
        self._relay_payload = ""
        self._relay_time = 0.0
        self._motor_errors: tuple[int, ...] = ()
        self._motor_time = 0.0
        self._mirror_errors: dict[str, str] = {}
        topics = system.observability.topics
        self._staged_joint_pub = rospy.Publisher(
            topics.staged_joint_state, JointState, queue_size=1
        )
        self._host_joint_pub = rospy.Publisher(
            topics.host_joint_state, JointState, queue_size=1
        )
        self._gripper_feedback_pub = rospy.Publisher(
            topics.gripper_feedback_state, JointState, queue_size=1
        )
        self._gripper_target_pub = rospy.Publisher(
            topics.gripper_target_state, JointState, queue_size=1
        )
        self._gripper_command_pub = rospy.Publisher(
            topics.gripper_command_state, JointState, queue_size=1
        )
        self._diagnostics_pub = rospy.Publisher(
            LEGACY_DIAGNOSTICS_TOPIC,
            DiagnosticArray,
            queue_size=1,
            latch=True,
        )
        primary = system.topics
        rospy.Subscriber(
            primary.staged_command,
            arm_control,
            self._staged_command_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.host_command,
            arm_control,
            self._host_command_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.gripper_feedback,
            JointState,
            self._gripper_feedback_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.gripper_target,
            gripper_position_control,
            self._gripper_target_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.gripper_command,
            gripper_position_control,
            self._gripper_command_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.relay_status,
            String,
            self._relay_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            primary.motor_status,
            status_stamped,
            self._motor_cb,
            queue_size=1,
        )

    def run(self) -> None:
        rate = rospy.Rate(self.system.observability.diagnostics_rate_hz)
        while not rospy.is_shutdown():
            self._publish_diagnostics()
            rate.sleep()

    def _staged_command_cb(self, message: arm_control) -> None:
        self._publish_arm_mirror(
            message,
            publisher=self._staged_joint_pub,
            label="staged joint command",
        )

    def _host_command_cb(self, message: arm_control) -> None:
        self._publish_arm_mirror(
            message,
            publisher=self._host_joint_pub,
            label="forwarded joint command",
        )

    def _publish_arm_mirror(self, message, *, publisher, label: str) -> None:
        joint_count = len(self.system.joint_safety.names)
        try:
            validate_arm_control_command(
                p_des=message.p_des,
                v_des=message.v_des,
                kp=message.kp,
                kd=message.kd,
                t_ff=message.t_ff,
                mode=message.mode,
                arm_joints=joint_count,
                allowed_modes=self.system.relay.allowed_control_modes,
            )
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            self._set_mirror_error(label, str(exc))
            return
        output = JointState()
        output.header = _source_header(message)
        output.name = list(self.system.joint_safety.names)
        output.position = [float(value) for value in message.p_des]
        publisher.publish(output)
        self._clear_mirror_error(label)

    def _gripper_feedback_cb(self, message: JointState) -> None:
        try:
            (stroke,) = require_finite_vector(
                message.position,
                count=1,
                label="gripper feedback",
            )
            self._validate_gripper_stroke(stroke)
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            self._set_mirror_error("gripper feedback", str(exc))
            return
        self._gripper_feedback_pub.publish(_gripper_joint_state(message, stroke))
        self._clear_mirror_error("gripper feedback")

    def _gripper_target_cb(self, message: gripper_position_control) -> None:
        self._publish_gripper_mirror(
            message,
            publisher=self._gripper_target_pub,
            label="gripper target",
        )

    def _gripper_command_cb(self, message: gripper_position_control) -> None:
        self._publish_gripper_mirror(
            message,
            publisher=self._gripper_command_pub,
            label="forwarded gripper command",
        )

    def _publish_gripper_mirror(self, message, *, publisher, label: str) -> None:
        try:
            stroke = float(message.gripper_stroke)
            self._validate_gripper_stroke(stroke)
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            self._set_mirror_error(label, str(exc))
            return
        publisher.publish(_gripper_joint_state(message, stroke))
        self._clear_mirror_error(label)

    def _validate_gripper_stroke(self, stroke: float) -> None:
        gripper = self.system.gripper
        reason = gripper_stroke_block_reason(
            stroke,
            minimum_mm=gripper.stroke_min_mm,
            maximum_mm=gripper.stroke_max_mm,
        )
        if reason is not None:
            raise ValueError(reason)

    def _relay_cb(self, message: String) -> None:
        with self._lock:
            self._relay_payload = str(message.data)
            self._relay_time = time.monotonic()

    def _motor_cb(self, message: status_stamped) -> None:
        try:
            errors = tuple(int(item.error_code) for item in message.data.motor_errors)
        except (AttributeError, OverflowError, TypeError, ValueError):
            errors = ()
        with self._lock:
            self._motor_errors = errors
            self._motor_time = time.monotonic()

    def _publish_diagnostics(self) -> None:
        with self._lock:
            relay_payload = self._relay_payload
            relay_time = self._relay_time
            motor_errors = self._motor_errors
            motor_time = self._motor_time
            mirror_errors = dict(self._mirror_errors)
        now = time.monotonic()
        if relay_time:
            relay = relay_diagnostic(
                relay_payload,
                age_s=freshness_age_s(relay_time, now_s=now),
                max_age_s=self.system.relay.max_status_age_s,
            )
        else:
            relay = DiagnosticFinding(
                name="A1/Relay",
                level=DIAGNOSTIC_ERROR,
                message="relay status unavailable",
            )
        if (
            not motor_time
            or freshness_age_s(motor_time, now_s=now)
            > self.system.relay.max_status_age_s
        ):
            motors = DiagnosticFinding(
                name="A1/Motors",
                level=DIAGNOSTIC_ERROR,
                message="motor status unavailable or stale",
            )
        else:
            motors = motor_diagnostic(
                motor_errors,
                arm_joints=len(self.system.joint_safety.names),
                gripper_ignored_error_mask=self.system.relay.gripper_ignored_error_mask,
            )
        telemetry = DiagnosticFinding(
            name="A1/Telemetry",
            level=DIAGNOSTIC_ERROR if mirror_errors else DIAGNOSTIC_OK,
            message=(
                "; ".join(
                    f"{key}: {value}" for key, value in sorted(mirror_errors.items())
                )
                if mirror_errors
                else "read-only telemetry mirrors healthy"
            ),
        )
        message = DiagnosticArray()
        message.header.stamp = rospy.Time.now()
        message.status = [
            _diagnostic_status(finding) for finding in (relay, motors, telemetry)
        ]
        self._diagnostics_pub.publish(message)

    def _set_mirror_error(self, label: str, error: str) -> None:
        with self._lock:
            self._mirror_errors[label] = error

    def _clear_mirror_error(self, label: str) -> None:
        with self._lock:
            self._mirror_errors.pop(label, None)


def _source_header(message):
    header = copy.deepcopy(getattr(message, "header", None))
    if header is None:
        output = JointState()
        header = output.header
    return header


def _gripper_joint_state(message, stroke: float) -> JointState:
    output = JointState()
    output.header = _source_header(message)
    output.name = ["gripper_stroke_mm"]
    output.position = [stroke]
    return output


def _diagnostic_status(finding: DiagnosticFinding) -> DiagnosticStatus:
    status = DiagnosticStatus()
    status.level = finding.level
    status.name = finding.name
    status.message = finding.message
    status.hardware_id = finding.hardware_id
    status.values = [KeyValue(key=key, value=value) for key, value in finding.values]
    return status


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / SYSTEM_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = load_system_config(args.config, repo_root=ROOT)
    if not system.observability.enabled:
        return 0
    rospy.init_node("a1_legacy_telemetry", anonymous=False)
    node = LegacyTelemetryNode(system)
    node.run()
    return 0
