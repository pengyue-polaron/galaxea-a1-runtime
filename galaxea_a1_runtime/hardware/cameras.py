"""Shared color camera helpers for A1 app scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealSenseColorCamera:
    """RealSense color stream wrapper with BGR/RGB read helpers."""

    def __init__(
        self,
        serial: str | None,
        width: int,
        height: int,
        fps: int,
        *,
        auto_exposure: bool = True,
        exposure: int = 140,
        gain: int = 32,
        auto_white_balance: bool = True,
        white_balance: int = 4600,
        warmup_frames: int = 30,
    ):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(cfg)
        _configure_realsense_color_sensor(
            profile,
            auto_exposure=auto_exposure,
            exposure=exposure,
            gain=gain,
            auto_white_balance=auto_white_balance,
            white_balance=white_balance,
        )
        for _ in range(max(0, warmup_frames)):
            self.pipeline.wait_for_frames()

    def read_bgr(self) -> np.ndarray | None:
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        frame = frames.get_color_frame()
        if not frame:
            return None
        return np.asanyarray(frame.get_data())

    def read_rgb(self) -> np.ndarray | None:
        bgr = self.read_bgr()
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.pipeline.stop()


class OpenCVColorCamera:
    """OpenCV/V4L color camera wrapper with explicit auto device resolution."""

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        *,
        backend_api: str = "v4l2",
        warmup_frames: int = 10,
    ):
        source, label = resolve_video_source(device)
        self.label = label
        api = cv2.CAP_V4L2 if backend_api == "v4l2" else 0
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device={device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(max(0, warmup_frames)):
            self.cap.read()

    def read_bgr(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def read_rgb(self) -> np.ndarray | None:
        bgr = self.read_bgr()
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


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


def _configure_realsense_color_sensor(
    profile: Any,
    *,
    auto_exposure: bool,
    exposure: int,
    gain: int,
    auto_white_balance: bool,
    white_balance: int,
) -> None:
    sensors = profile.get_device().query_sensors()
    if len(sensors) < 2:
        return
    color_sensor = sensors[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
    if not auto_exposure:
        color_sensor.set_option(rs.option.exposure, float(exposure))
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0)
    if not auto_white_balance and color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))
