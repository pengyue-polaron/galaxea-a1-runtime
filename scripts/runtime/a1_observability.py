#!/usr/bin/env python3.12
# ruff: noqa: E402
"""Publish A1 telemetry and proxy narrowly scoped Operator Session controls."""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.configuration.system import (  # noqa: E402
    SYSTEM_CONFIG,
    SystemConfig,
    load_system_config,
)
from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from embodied_ops.foxglove import (  # noqa: E402
    foxglove_workflow_status,
    prepare_collection_action,
    prepare_collection_stop,
)
from galaxea_a1_runtime.runtime.operator_session import (  # noqa: E402
    OperatorSessionClient,
    OperatorSessionUnavailable,
)
from galaxea_a1_runtime.observability import (  # noqa: E402
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_OK,
    DiagnosticFinding,
    camera_diagnostic,
    collection_action_service_bindings,
    motor_diagnostic,
    operator_panel_diagnostic,
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
from sensor_msgs.msg import CompressedImage, JointState
from signal_arm.msg import arm_control, gripper_position_control, status_stamped
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeReaders  # noqa: E402
from galaxea_a1_runtime.hardware.cameras import RealSenseFrameSet  # noqa: E402


class A1ObservabilityNode:
    def __init__(self, system: SystemConfig) -> None:
        self.system = system
        self._lock = threading.RLock()
        self._relay_payload = ""
        self._relay_time = 0.0
        self._motor_errors: tuple[int, ...] = ()
        self._motor_time = 0.0
        self._mirror_errors: dict[str, str] = {}
        self._camera: CameraBridgeReaders | None = None
        self._camera_error = "Camera Bridge not connected"
        self._camera_retry_at = 0.0
        self._camera_ages: tuple[float | None, float | None, float | None] = (
            None,
            None,
            None,
        )
        self._last_camera_pair = (-1, -1)
        self._operator_telemetry = foxglove_workflow_status(None)
        self._operator_session = OperatorSessionClient(
            timeout_s=system.observability.operator_session_timeout_s
        )
        self._operator_services = []
        topics = system.observability.topics
        self._front_image_pub = rospy.Publisher(
            topics.front_image, CompressedImage, queue_size=1
        )
        self._wrist_image_pub = rospy.Publisher(
            topics.wrist_image, CompressedImage, queue_size=1
        )
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
        self._workflow_status_pub = rospy.Publisher(
            topics.workflow_status,
            String,
            queue_size=1,
            latch=True,
        )
        self._diagnostics_pub = rospy.Publisher(
            topics.diagnostics,
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
        if system.operator_panel.control_enabled:
            self._advertise_operator_services()
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

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()

    def run(self) -> None:
        observation = self.system.observability
        loop_rate = rospy.Rate(
            max(
                observation.image_rate_hz,
                observation.diagnostics_rate_hz,
                observation.operator_panel_poll_rate_hz,
            )
        )
        image_period = 1.0 / observation.image_rate_hz
        diagnostics_period = 1.0 / observation.diagnostics_rate_hz
        operator_panel_period = 1.0 / observation.operator_panel_poll_rate_hz
        next_image = 0.0
        next_diagnostics = 0.0
        next_operator_panel = 0.0
        while not rospy.is_shutdown():
            now = time.monotonic()
            if now >= next_image:
                self._publish_camera_pair(now)
                next_image = now + image_period
            if now >= next_diagnostics:
                self._publish_diagnostics(now)
                next_diagnostics = now + diagnostics_period
            if now >= next_operator_panel:
                self._poll_operator_panel()
                next_operator_panel = now + operator_panel_period
            loop_rate.sleep()

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
        output.header = _fresh_header(message)
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

    def _publish_camera_pair(self, now: float) -> None:
        camera = self._camera
        if camera is None:
            if now >= self._camera_retry_at:
                self._connect_camera(now)
            return
        error = camera.exception()
        if error is not None:
            self._disconnect_camera(now, f"Camera Bridge failed: {error}")
            return
        pair = camera.latest_pair()
        if pair is None:
            return
        front, wrist = pair
        front_age = max(0.0, now - front.monotonic_s)
        wrist_age = max(0.0, now - wrist.monotonic_s)
        skew = abs(front.monotonic_s - wrist.monotonic_s)
        self._camera_ages = (front_age, wrist_age, skew)
        cameras = self.system.cameras
        if (
            front_age > cameras.max_age_s
            or wrist_age > cameras.max_age_s
            or skew > cameras.max_pair_skew_s
        ):
            return
        pair_id = (front.seq, wrist.seq)
        if pair_id == self._last_camera_pair:
            return
        front_bgr = (
            front.value.color_bgr
            if isinstance(front.value, RealSenseFrameSet)
            else front.value
        )
        stamp = rospy.Time.now()
        try:
            front_message = _compressed_image(
                front_bgr,
                stamp=stamp,
                jpeg_quality=self.system.observability.jpeg_quality,
            )
            wrist_message = _compressed_image(
                wrist.value,
                stamp=stamp,
                jpeg_quality=self.system.observability.jpeg_quality,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._camera_error = f"camera JPEG encoding failed: {exc}"
            return
        self._front_image_pub.publish(front_message)
        self._wrist_image_pub.publish(wrist_message)
        self._last_camera_pair = pair_id
        self._camera_error = ""

    def _connect_camera(self, now: float) -> None:
        camera = CameraBridgeReaders(self.system.cameras)
        try:
            camera.start(timeout_s=self.system.observability.camera_connect_timeout_s)
        except (OSError, RuntimeError, ValueError) as exc:
            camera.close()
            self._camera_error = f"Camera Bridge unavailable: {exc}"
            self._camera_retry_at = now + self.system.observability.camera_retry_s
            return
        self._camera = camera
        self._camera_error = ""

    def _disconnect_camera(self, now: float, reason: str) -> None:
        camera = self._camera
        self._camera = None
        if camera is not None:
            try:
                camera.close()
            except RuntimeError as exc:
                reason = f"{reason}; close failed: {exc}"
        self._camera_error = reason
        self._camera_retry_at = now + self.system.observability.camera_retry_s
        self._camera_ages = (None, None, None)

    def _publish_diagnostics(self, now: float) -> None:
        with self._lock:
            relay_payload = self._relay_payload
            relay_time = self._relay_time
            motor_errors = self._motor_errors
            motor_time = self._motor_time
            mirror_errors = dict(self._mirror_errors)
        if relay_time:
            relay = relay_diagnostic(
                relay_payload,
                age_s=now - relay_time,
                max_age_s=self.system.relay.max_status_age_s,
            )
        else:
            relay = DiagnosticFinding(
                name="A1/Relay",
                level=DIAGNOSTIC_ERROR,
                message="relay status unavailable",
            )
        if not motor_time or now - motor_time > self.system.relay.max_status_age_s:
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
        front_age, wrist_age, skew = self._camera_ages
        cameras = camera_diagnostic(
            connected=self._camera is not None,
            front_age_s=front_age,
            wrist_age_s=wrist_age,
            pair_skew_s=skew,
            max_age_s=self.system.cameras.max_age_s,
            max_pair_skew_s=self.system.cameras.max_pair_skew_s,
            error=self._camera_error,
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
        operator_panel = operator_panel_diagnostic(self._operator_telemetry)
        message = DiagnosticArray()
        message.header.stamp = rospy.Time.now()
        message.status = [
            _diagnostic_status(finding)
            for finding in (relay, motors, cameras, telemetry, operator_panel)
        ]
        self._diagnostics_pub.publish(message)

    def _poll_operator_panel(self) -> None:
        try:
            normalized = foxglove_workflow_status(self._operator_session.status())
        except (OperatorSessionUnavailable, RuntimeError, ValueError) as exc:
            normalized = foxglove_workflow_status(
                None,
                error=f"Operator Session unavailable: {exc}",
            )
        payload = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self._operator_telemetry = normalized
        self._workflow_status_pub.publish(String(data=payload))

    def _advertise_operator_services(self) -> None:
        services = self.system.operator_panel.services
        for binding in collection_action_service_bindings(self.system):
            self._operator_services.append(
                rospy.Service(
                    binding.service_name,
                    Trigger,
                    self._operator_action_handler(
                        binding.action_id, binding.expected_phase
                    ),
                )
            )
        self._operator_services.append(
            rospy.Service(services.stop, Trigger, self._operator_stop_handler)
        )

    def _operator_action_handler(self, action_id: str, expected_phase: str):
        def handle(_request) -> TriggerResponse:
            try:
                action = prepare_collection_action(
                    self._operator_session.status(),
                    action_id=action_id,
                    expected_phase=expected_phase,
                )
                self._operator_session.input(
                    action.action_id,
                    run_id=action.run_id,
                    input_revision=action.input_revision,
                )
            except (OperatorSessionUnavailable, RuntimeError, ValueError) as exc:
                return TriggerResponse(success=False, message=str(exc))
            return TriggerResponse(
                success=True,
                message=f"accepted {action_id} for run {action.run_id}",
            )

        return handle

    def _operator_stop_handler(self, _request) -> TriggerResponse:
        try:
            run_id = prepare_collection_stop(self._operator_session.status())
            self._operator_session.stop(run_id=run_id)
        except (OperatorSessionUnavailable, RuntimeError, ValueError) as exc:
            return TriggerResponse(success=False, message=str(exc))
        return TriggerResponse(
            success=True, message=f"stopping collection run {run_id}"
        )

    def _set_mirror_error(self, label: str, error: str) -> None:
        with self._lock:
            self._mirror_errors[label] = error

    def _clear_mirror_error(self, label: str) -> None:
        with self._lock:
            self._mirror_errors.pop(label, None)


def _fresh_header(message):
    header = copy.deepcopy(getattr(message, "header", None))
    if header is None:
        output = JointState()
        header = output.header
    header.stamp = rospy.Time.now()
    return header


def _gripper_joint_state(message, stroke: float) -> JointState:
    output = JointState()
    output.header = _fresh_header(message)
    output.name = ["gripper_stroke_mm"]
    output.position = [stroke]
    return output


def _compressed_image(image, *, stamp, jpeg_quality: int) -> CompressedImage:
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            f"camera image must be HxWx3 uint8 BGR, got shape={array.shape}, dtype={array.dtype}"
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        array,
        (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality),
    )
    if not ok:
        raise RuntimeError("OpenCV did not encode the image")
    message = CompressedImage()
    message.header.stamp = stamp
    message.format = "jpeg"
    message.data = encoded.tobytes()
    return message


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
    rospy.init_node("a1_observability", anonymous=False)
    node = A1ObservabilityNode(system)
    rospy.on_shutdown(node.close)
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
