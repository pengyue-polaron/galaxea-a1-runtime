"""ROS 2 driver composition and timestamp-based observation ownership.

The Unix endpoint is a boundary for isolated model/ROS 1 environments, not a
second acquisition or synchronization implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.hardware.camera_bridge import (
    CameraBridgeReader,
    CameraBridgeServer,
)
from galaxea_a1_runtime.hardware.camera_reader import CameraSample
from galaxea_a1_runtime.hardware.cameras import RealSenseFrameSet
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
)

ROS2_IMAGE = "galaxea-a1-runtime/ros2-jazzy:local"
ROS2_CAMERA_CONTAINER = "a1-ros2-cameras"
CAMERA_NAMESPACE = "/a1/cameras"
SYNC_DIAGNOSTICS_TOPIC = f"{CAMERA_NAMESPACE}/sync_status"


def image_topic(name: str) -> str:
    return f"{CAMERA_NAMESPACE}/{name}/color/image_raw"


def driver_parameters(camera: Any, color_module: str) -> dict[str, Any]:
    """Translate the sole tracked camera contract into official driver parameters."""
    params = {
        "serial_no": f"_{camera.serial}",
        "enable_color": True,
        "enable_depth": camera.depth,
        "enable_infra1": False,
        "enable_infra2": False,
        "enable_gyro": False,
        "enable_accel": False,
        f"{color_module}.color_profile": f"{camera.width},{camera.height},{camera.fps}",
        f"{color_module}.color_format": "RGB8",
        f"{color_module}.enable_auto_exposure": camera.auto_exposure,
        f"{color_module}.enable_auto_white_balance": camera.auto_white_balance,
        f"{color_module}.global_time_enabled": True,
        # Reliable writers allow rosbag2 to retain raw images. The live
        # synchronizer requests sensor QoS to avoid waiting for old frames.
        "color_qos": "DEFAULT",
        "depth_qos": "DEFAULT",
        "diagnostics_period": 1.0,
        "enable_sync": camera.depth,
        "align_depth.enable": camera.depth,
    }
    if not camera.auto_exposure:
        params.update(
            {
                f"{color_module}.exposure": camera.exposure,
                f"{color_module}.gain": camera.gain,
            }
        )
    if not camera.auto_white_balance:
        params[f"{color_module}.white_balance"] = camera.white_balance
    if camera.depth:
        params["depth_module.depth_profile"] = (
            f"{camera.depth_width},{camera.depth_height},{camera.fps}"
        )
        params["depth_module.global_time_enabled"] = True
    return params


class SynchronizedCameras:
    """One atomic cache filled exclusively by upstream ApproximateTimeSynchronizer."""

    def __init__(self, node: Any, config: Any):
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, SimpleFilter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.node = node
        self.config = config
        self._lock = threading.Lock()
        self._pair: tuple[CameraSample, CameraSample] | None = None
        self._error: BaseException | None = None
        self.front = CameraBridgeReader(self, "front")
        self.wrist = CameraBridgeReader(self, "wrist")
        self._cv = CvBridge()
        self._offset = time.time() - time.perf_counter()
        self._last_stamp: dict[str, int] = {}
        self.received: dict[str, int] = {}
        self.rejected = 0
        self.pairs = 0
        self.last_reason = "waiting_for_frames"
        self._filters: list[Any] = []
        self._subscriptions: list[Any] = []
        self._status_type = String
        self._status = node.create_publisher(String, SYNC_DIAGNOSTICS_TOPIC, 1)
        streams = [("front", image_topic("front")), ("wrist", image_topic("wrist"))]
        if config.front.depth:
            streams.append(
                ("depth", f"{CAMERA_NAMESPACE}/front/aligned_depth_to_color/image_raw")
            )
        for name, topic in streams:
            filtered = SimpleFilter()
            self._filters.append(filtered)
            self._subscriptions.append(
                node.create_subscription(
                    Image,
                    topic,
                    lambda message, key=name, target=filtered: self._receive(
                        key, target, message
                    ),
                    qos_profile_sensor_data,
                )
            )
        # Cache at most one freshness window, not an unbounded queue.
        queue_size = max(
            2, math.ceil(max(config.front.fps, config.wrist.fps) * config.max_age_s)
        )
        self.synchronizer = ApproximateTimeSynchronizer(
            self._filters,
            queue_size=queue_size,
            slop=config.max_pair_skew_s,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self._synchronized)
        self._timer = node.create_timer(1.0, self._publish_status)

    @staticmethod
    def stamp_ns(message: Any) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )

    def _receive(self, name: str, target: Any, message: Any) -> None:
        stamp = self.stamp_ns(message)
        now = time.time()
        self.received[name] = self.received.get(name, 0) + 1
        age = now - stamp / 1e9
        if (
            stamp <= self._last_stamp.get(name, 0)
            or not 0 <= age <= self.config.max_age_s
        ):
            self.rejected += 1
            self.last_reason = (
                f"{name}: non-increasing, future or stale source timestamp"
            )
            return
        self._last_stamp[name] = stamp
        if self.received[name] <= self.config.warmup_frames:
            return
        target.signalMessage(message)

    def _synchronized(self, front: Any, wrist: Any, depth: Any = None) -> None:
        try:
            now = time.perf_counter()
            # A wall-clock step must never turn old images into fresh observations.
            if abs(time.time() - now - self._offset) > self.config.max_pair_skew_s:
                raise RuntimeError(
                    "system clock stepped; restart camera synchronization"
                )
            stamps = [self.stamp_ns(front), self.stamp_ns(wrist)]
            if depth is not None:
                stamps.append(self.stamp_ns(depth))
            times = [stamp / 1e9 - self._offset for stamp in stamps]
            if any(not 0 <= now - stamp <= self.config.max_age_s for stamp in times):
                self.rejected += 1
                self.last_reason = "matched pair expired while queued"
                return
            front_image = self._cv.imgmsg_to_cv2(front, desired_encoding="bgr8").copy()
            wrist_image = self._cv.imgmsg_to_cv2(wrist, desired_encoding="bgr8").copy()
            depth_image = (
                None
                if depth is None
                else self._cv.imgmsg_to_cv2(depth, desired_encoding="16UC1").copy()
            )
            for image, camera in (
                (front_image, self.config.front),
                (wrist_image, self.config.wrist),
            ):
                if image.shape != (camera.height, camera.width, 3):
                    raise ValueError(
                        "ROS camera image dimensions differ from System config"
                    )
            with self._lock:
                seq = self.pairs
                self._pair = (
                    CameraSample(
                        seq,
                        times[0],
                        RealSenseFrameSet(front_image, depth_image),
                        stamps[0],
                        "ros_system_time",
                    ),
                    CameraSample(
                        seq, times[1], wrist_image, stamps[1], "ros_system_time"
                    ),
                )
                self.pairs += 1
                self.last_reason = "ok"
        except BaseException as exc:
            with self._lock:
                self._error = exc

    def latest_pair(self) -> tuple[CameraSample, CameraSample] | None:
        with self._lock:
            return self._pair

    def _latest_sample(self, name: str) -> CameraSample | None:
        pair = self.latest_pair()
        return None if pair is None else pair[0 if name == "front" else 1]

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._error

    def _publish_status(self) -> None:
        pair = self.latest_pair()
        message = self._status_type()
        message.data = json.dumps(
            {
                "received": self.received,
                "matched_pairs": self.pairs,
                "rejected_inputs": self.rejected,
                "reason": self.last_reason,
                "source_clock": "ros_system_time",
                "pair_skew_s": None
                if pair is None
                else abs(pair[0].source_stamp_ns - pair[1].source_stamp_ns) / 1e9,
                "oldest_age_s": None
                if pair is None
                else time.perf_counter() - min(p.monotonic_s for p in pair),
            }
        )
        self._status.publish(message)


def main() -> int:
    import rclpy
    from rclpy.node import Node

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--front-usb-type", required=True)
    args = parser.parse_args()
    system = load_system_config(args.config)
    rclpy.init()
    node = Node("a1_camera_sync")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    cameras = SynchronizedCameras(node, system.cameras)
    bridge = CameraBridgeServer(
        system.cameras,
        front_reader=cameras.front,
        wrist_reader=cameras.wrist,
        read_pair=cameras.latest_pair,
        front_source=f"ros2:realsense:{system.cameras.front.serial}",
        wrist_source=f"ros2:realsense:{system.cameras.wrist.serial}",
        front_usb_type=args.front_usb_type,
    )
    preview = CameraWebPreview(
        replace(system.web_preview, enabled=True),
        max_source_age_s=system.cameras.max_age_s,
    )
    preview.register_reader(
        "agent",
        cameras.front,
        extract=color_from_frameset,
        source=bridge.metadata.front_source,
    )
    preview.register_reader(
        "wrist",
        cameras.wrist,
        extract=color_from_bgr,
        source=bridge.metadata.wrist_source,
    )
    try:
        bridge.start()
        preview.start()
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            error = cameras.exception() or bridge.exception()
            if error is not None:
                raise RuntimeError("ROS 2 camera pipeline failed") from error
    finally:
        preview.close()
        bridge.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
