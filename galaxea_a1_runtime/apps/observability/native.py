"""Native ROS 2 display and guarded Operator Session services."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger
from embodied_ops.foxglove import (
    foxglove_workflow_status,
    prepare_collection_action,
    prepare_collection_stop,
)
from galaxea_a1_runtime.configuration.system import load_system_config, SystemConfig
from galaxea_a1_runtime.runtime.operator_session import (
    OperatorSessionClient,
    OperatorSessionUnavailable,
)
from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeReaders
from galaxea_a1_runtime.hardware.cameras import RealSenseFrameSet
from galaxea_a1_runtime.runtime.ros2 import LEGACY_DIAGNOSTICS_TOPIC
from galaxea_a1_runtime.observability import (
    DiagnosticFinding,
    DIAGNOSTIC_ERROR,
    camera_diagnostic,
    operator_panel_diagnostic,
    collection_action_service_bindings,
)


class ObservabilityNode(Node):
    def __init__(self, system: SystemConfig) -> None:
        super().__init__(
            "a1_observability", enable_rosout=True, start_parameter_services=False
        )
        self.system = system
        self._camera = None
        self._camera_error = "Camera Bridge not connected"
        self._camera_retry_at = 0.0
        self._camera_ages = (None, None, None)
        self._last_camera_pair = (-1, -1)
        self._operator_telemetry = foxglove_workflow_status(None)
        self._operator_session = OperatorSessionClient(
            timeout_s=system.observability.operator_session_timeout_s
        )
        self._operator_services = []
        self._legacy_status = []
        self._legacy_received_at = 0.0
        topics = system.observability.topics
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._front_image_pub = self.create_publisher(
            CompressedImage, topics.front_image, 1
        )
        self._wrist_image_pub = self.create_publisher(
            CompressedImage, topics.wrist_image, 1
        )
        self._workflow_status_pub = self.create_publisher(
            String, topics.workflow_status, retained
        )
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray, topics.diagnostics, retained
        )
        self.create_subscription(
            DiagnosticArray, LEGACY_DIAGNOSTICS_TOPIC, self._legacy_cb, 1
        )
        if system.operator_panel.control_enabled:
            self._advertise_operator_services()
        observation = system.observability
        self.create_timer(
            1.0 / observation.image_rate_hz,
            lambda: self._publish_camera_pair(time.monotonic()),
        )
        self.create_timer(
            1.0 / observation.diagnostics_rate_hz, self._publish_diagnostics
        )
        self.create_timer(
            1.0 / observation.operator_panel_poll_rate_hz, self._poll_operator_panel
        )

    def _legacy_cb(self, message):
        self._legacy_status = message.status
        self._legacy_received_at = time.monotonic()

    def _publish_diagnostics(self):
        now = time.monotonic()
        # Both the ROS 1 adapter and its one-way bridge must remain alive.
        if (
            now - self._legacy_received_at
            > 2.0 / self.system.observability.diagnostics_rate_hz
        ):
            legacy = [
                _diagnostic_status(
                    DiagnosticFinding(
                        name="A1/Legacy bridge",
                        level=DIAGNOSTIC_ERROR,
                        message="ROS 1 telemetry unavailable or stale",
                    )
                )
            ]
        else:
            legacy = self._legacy_status
        front_age, wrist_age, skew = self._camera_ages
        camera = camera_diagnostic(
            connected=self._camera is not None,
            front_age_s=front_age,
            wrist_age_s=wrist_age,
            pair_skew_s=skew,
            max_age_s=self.system.cameras.max_age_s,
            max_pair_skew_s=self.system.cameras.max_pair_skew_s,
            error=self._camera_error,
        )
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [
            *legacy,
            _diagnostic_status(camera),
            _diagnostic_status(operator_panel_diagnostic(self._operator_telemetry)),
        ]
        self._diagnostics_pub.publish(message)

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()

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
            self._camera_ages = (None, None, None)
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
        try:
            front_message = _compressed_image(
                front_bgr,
                stamp=Time(
                    sec=front.source_stamp_ns // 1_000_000_000,
                    nanosec=front.source_stamp_ns % 1_000_000_000,
                ),
                jpeg_quality=self.system.observability.jpeg_quality,
            )
            wrist_message = _compressed_image(
                wrist.value,
                stamp=Time(
                    sec=wrist.source_stamp_ns // 1_000_000_000,
                    nanosec=wrist.source_stamp_ns % 1_000_000_000,
                ),
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
        self._last_camera_pair = (-1, -1)
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
                self.create_service(
                    Trigger,
                    binding.service_name,
                    self._operator_action_handler(
                        binding.action_id, binding.expected_phase
                    ),
                )
            )
        self._operator_services.append(
            self.create_service(Trigger, services.stop, self._operator_stop_handler)
        )

    def _operator_action_handler(self, action_id: str, expected_phase: str):
        def handle(_request, _response) -> Trigger.Response:
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
                return Trigger.Response(success=False, message=str(exc))
            return Trigger.Response(
                success=True,
                message=f"accepted {action_id} for run {action.run_id}",
            )

        return handle

    def _operator_stop_handler(self, _request, _response) -> Trigger.Response:
        try:
            run_id = prepare_collection_stop(self._operator_session.status())
            self._operator_session.stop(run_id=run_id)
        except (OperatorSessionUnavailable, RuntimeError, ValueError) as exc:
            return Trigger.Response(success=False, message=str(exc))
        return Trigger.Response(
            success=True, message=f"stopping collection run {run_id}"
        )


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
    status.level = bytes((finding.level,))
    status.name = finding.name
    status.message = finding.message
    status.hardware_id = finding.hardware_id
    status.values = [KeyValue(key=key, value=value) for key, value in finding.values]
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    system = load_system_config(args.config)
    rclpy.init()
    node = ObservabilityNode(system)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
