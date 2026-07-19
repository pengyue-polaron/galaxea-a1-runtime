"""Shared color camera helpers for A1 app scripts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.system import (
    SystemCameraDeviceConfig,
    SystemRealSenseCameraConfig,
    SystemV4l2CameraConfig,
)
from galaxea_a1_runtime.hardware.camera_reader import (
    CameraReader,
    CameraSample,
    LatestCameraReader,
    close_camera_resources,
)

__all__ = [
    "CameraSample",
    "CameraReader",
    "ColorCamera",
    "LatestCameraReader",
    "OpenCVColorCamera",
    "RealSenseColorCamera",
    "RealSenseDeviceInfo",
    "RealSenseFrameSet",
    "close_camera_resources",
    "open_configured_camera",
    "realsense_device_info",
    "realsense_usb_is_superspeed",
    "resolve_video_source",
    "video_device_name",
]

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

REALSENSE_FRAME_TIMEOUT_MS = 1000
REALSENSE_WARMUP_MARGIN_S = 5.0


@dataclass(frozen=True)
class RealSenseFrameSet:
    color_bgr: np.ndarray
    depth_mm: np.ndarray | None = None


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    name: str
    serial: str
    usb_type: str


class RealSenseColorCamera:
    """RealSense color stream wrapper with optional aligned depth frames."""

    def __init__(
        self,
        serial: str | None,
        width: int,
        height: int,
        fps: int,
        *,
        auto_exposure: bool,
        exposure: int | None,
        gain: int | None,
        auto_white_balance: bool,
        white_balance: int | None,
        enable_depth: bool,
        depth_width: int | None,
        depth_height: int | None,
        align_depth_to_color: bool | None,
        warmup_frames: int,
        require_usb3: bool,
    ):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")
        if min(width, height, fps) <= 0:
            raise ValueError("RealSense color dimensions and fps must be positive")
        if warmup_frames < 0:
            raise ValueError("RealSense warmup_frames must be non-negative")
        if not auto_exposure and (exposure is None or gain is None):
            raise ValueError("manual RealSense exposure and gain are required")
        if not auto_white_balance and white_balance is None:
            raise ValueError("manual RealSense white balance is required")
        if enable_depth and (
            depth_width is None or depth_height is None or align_depth_to_color is None
        ):
            raise ValueError("enabled RealSense depth settings are incomplete")
        info = realsense_device_info(serial)
        if info is None:
            raise RuntimeError("No RealSense device found")
        self.serial = info.serial
        self.usb_type = info.usb_type
        self.label = f"realsense:{info.name}:{info.serial}"
        if require_usb3 and not realsense_usb_is_superspeed(info.usb_type):
            raise RuntimeError(
                "RealSense is enumerated as USB "
                f"{info.usb_type or 'unknown'}, but this config requires USB3. "
                "Plug the RealSense into a SuperSpeed USB3 port/cable, or lower "
                "the tracked front camera config before recording."
            )
        self._enable_depth = enable_depth
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if info.serial:
            cfg.enable_device(info.serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            assert depth_width is not None and depth_height is not None
            cfg.enable_stream(
                rs.stream.depth,
                depth_width,
                depth_height,
                rs.format.z16,
                fps,
            )
        try:
            profile = self.pipeline.start(cfg)
            self._align = (
                rs.align(rs.stream.color)
                if enable_depth and align_depth_to_color
                else None
            )
            _configure_realsense_color_sensor(
                profile,
                auto_exposure=auto_exposure,
                exposure=exposure,
                gain=gain,
                auto_white_balance=auto_white_balance,
                white_balance=white_balance,
            )
            warmup_timeout_s = max(
                REALSENSE_WARMUP_MARGIN_S,
                float(warmup_frames) / float(fps) + REALSENSE_WARMUP_MARGIN_S,
            )
            self._warmup(warmup_frames, timeout_s=warmup_timeout_s)
        except BaseException:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise

    def read_frameset(self) -> RealSenseFrameSet | None:
        frames = self.pipeline.poll_for_frames()
        return self._decode_frames(frames) if frames else None

    def wait_frameset(
        self, *, timeout_ms: int = REALSENSE_FRAME_TIMEOUT_MS
    ) -> RealSenseFrameSet | None:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None
        return self._decode_frames(frames)

    def _decode_frames(self, frames: Any) -> RealSenseFrameSet | None:
        if not frames:
            return None
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        depth_mm = None
        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_mm = np.asanyarray(depth_frame.get_data()).astype(
                np.uint16, copy=False
            )
        return RealSenseFrameSet(
            color_bgr=np.asanyarray(color_frame.get_data()),
            depth_mm=depth_mm,
        )

    def read_bgr(self) -> np.ndarray | None:
        frameset = self.read_frameset()
        return None if frameset is None else frameset.color_bgr

    def read_rgb(self) -> np.ndarray | None:
        bgr = self.read_bgr()
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_depth_mm(self) -> np.ndarray | None:
        frameset = self.read_frameset()
        return None if frameset is None else frameset.depth_mm

    def close(self) -> None:
        self.pipeline.stop()

    def _warmup(self, warmup_frames: int, *, timeout_s: float) -> None:
        if warmup_frames <= 0:
            return
        deadline = time.monotonic() + timeout_s
        frames = 0
        timeouts = 0
        while frames < warmup_frames and time.monotonic() < deadline:
            try:
                frame = self.pipeline.wait_for_frames(REALSENSE_FRAME_TIMEOUT_MS)
            except RuntimeError:
                timeouts += 1
                continue
            if frame:
                frames += 1
        if frames < warmup_frames:
            raise RuntimeError(
                "RealSense produced "
                f"{frames}/{warmup_frames} warmup frames in {timeout_s:.1f}s "
                f"(timeouts={timeouts}, usb={self.usb_type or 'unknown'}). "
                "Check the USB3 connection and camera bandwidth before recording."
            )


class OpenCVColorCamera:
    """OpenCV/V4L color camera wrapper with explicit auto device resolution."""

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        *,
        backend_api: str,
        pixel_format: str,
        warmup_frames: int,
    ):
        if min(width, height, fps) <= 0:
            raise ValueError("V4L2 camera dimensions and fps must be positive")
        if warmup_frames < 0:
            raise ValueError("V4L2 warmup_frames must be non-negative")
        source, label = resolve_video_source(device)
        self.label = label
        api = cv2.CAP_V4L2 if backend_api == "v4l2" else 0
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device={device}")
        if pixel_format:
            if len(pixel_format) != 4:
                raise ValueError(
                    f"pixel_format must be a four-character code, got {pixel_format!r}"
                )
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(warmup_frames):
            self.cap.read()

    def read_bgr(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def read_rgb(self) -> np.ndarray | None:
        bgr = self.read_bgr()
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


ColorCamera = RealSenseColorCamera | OpenCVColorCamera


def open_configured_camera(
    config: SystemCameraDeviceConfig,
    *,
    warmup_frames: int,
    enable_depth: bool,
) -> ColorCamera:
    """Open one camera with every physical setting from the system config."""

    if isinstance(config, SystemRealSenseCameraConfig):
        if enable_depth and not config.depth:
            raise ValueError("depth was requested but is disabled in system config")
        return RealSenseColorCamera(
            config.serial,
            config.width,
            config.height,
            config.fps,
            auto_exposure=config.auto_exposure,
            exposure=config.exposure,
            gain=config.gain,
            auto_white_balance=config.auto_white_balance,
            white_balance=config.white_balance,
            warmup_frames=warmup_frames,
            require_usb3=config.require_usb3,
            enable_depth=enable_depth,
            depth_width=config.depth_width if enable_depth else None,
            depth_height=config.depth_height if enable_depth else None,
            align_depth_to_color=(
                config.align_depth_to_color if enable_depth else None
            ),
        )
    if enable_depth:
        raise ValueError("V4L2 camera config does not support depth")
    if isinstance(config, SystemV4l2CameraConfig):
        return OpenCVColorCamera(
            config.device,
            config.width,
            config.height,
            config.fps,
            backend_api=config.backend_api,
            pixel_format=config.pixel_format,
            warmup_frames=warmup_frames,
        )
    raise TypeError(f"unsupported camera config type: {type(config).__name__}")


def resolve_video_source(device: str) -> tuple[int | str, str]:
    """Resolve a configured OpenCV device.

    The special value `auto` selects the first readable non-RealSense V4L camera
    so the wrist camera does not accidentally bind to the front RealSense stream.
    """

    if device.strip().lower() != "auto":
        source: int | str = int(device) if device.isdigit() else device
        return source, str(device)
    for index in range(16):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if not ok:
            continue
        name = video_device_name(index)
        if "realsense" in name.lower() or "intel" in name.lower():
            continue
        return index, f"video{index}:{name}"
    raise RuntimeError("No suitable wrist camera found")


def video_device_name(index: int) -> str:
    path = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        return path.read_text().strip() if path.is_file() else "unknown"
    except OSError:
        return "unknown"


def realsense_device_info(serial: str | None = None) -> RealSenseDeviceInfo | None:
    if rs is None:
        return None
    devices = list(rs.context().query_devices())
    if serial:
        for device in devices:
            if _rs_device_info(device, rs.camera_info.serial_number) == serial:
                return _realsense_device_info_from_device(device)
        found = ", ".join(
            _rs_device_info(device, rs.camera_info.serial_number) for device in devices
        )
        raise RuntimeError(f"RealSense serial {serial!r} not found; found [{found}]")
    if not devices:
        return None
    return _realsense_device_info_from_device(devices[0])


def realsense_usb_is_superspeed(usb_type: str) -> bool:
    try:
        return float(usb_type) >= 3.0
    except (TypeError, ValueError):
        return False


def _realsense_device_info_from_device(device: Any) -> RealSenseDeviceInfo:
    return RealSenseDeviceInfo(
        name=_rs_device_info(device, rs.camera_info.name),
        serial=_rs_device_info(device, rs.camera_info.serial_number),
        usb_type=_rs_device_info(device, rs.camera_info.usb_type_descriptor),
    )


def _rs_device_info(device: Any, key: Any) -> str:
    try:
        return str(device.get_info(key))
    except Exception:
        return ""


def _configure_realsense_color_sensor(
    profile: Any,
    *,
    auto_exposure: bool,
    exposure: int | None,
    gain: int | None,
    auto_white_balance: bool,
    white_balance: int | None,
) -> None:
    sensors = profile.get_device().query_sensors()
    color_sensor = _find_realsense_color_sensor(sensors)
    if color_sensor is None:
        return
    color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
    if not auto_exposure:
        assert exposure is not None and gain is not None
        color_sensor.set_option(rs.option.exposure, float(exposure))
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(
            rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0
        )
    if not auto_white_balance and color_sensor.supports(rs.option.white_balance):
        assert white_balance is not None
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def _find_realsense_color_sensor(sensors: Any) -> Any | None:
    for sensor in sensors:
        if sensor.supports(rs.option.enable_auto_white_balance):
            return sensor
    if len(sensors) >= 2:
        return sensors[1]
    return None
