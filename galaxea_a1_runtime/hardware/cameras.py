"""Shared color camera helpers for A1 app scripts."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass(frozen=True)
class RealSenseFrameSet:
    color_bgr: np.ndarray
    depth_mm: np.ndarray | None = None


@dataclass(frozen=True)
class CameraSample:
    seq: int
    monotonic_s: float
    value: Any


class LatestCameraReader:
    """Continuously read one camera and expose the newest successful sample."""

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], Any | None],
        *,
        idle_sleep_s: float = 0.002,
    ):
        self.name = name
        self._read_fn = read_fn
        self._idle_sleep_s = idle_sleep_s
        self._lock = threading.Lock()
        self._latest: CameraSample | None = None
        self._exception: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{name}-camera-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> CameraSample | None:
        with self._lock:
            return self._latest

    def latest_seq(self) -> int:
        latest = self.latest()
        return -1 if latest is None else latest.seq

    def frame_count(self) -> int:
        return self.latest_seq() + 1

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._exception

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            try:
                value = self._read_fn()
            except BaseException as exc:  # noqa: BLE001 - cross-thread surfacing.
                with self._lock:
                    self._exception = exc
                return
            if value is None:
                time.sleep(self._idle_sleep_s)
                continue
            with self._lock:
                self._latest = CameraSample(seq=seq, monotonic_s=time.perf_counter(), value=value)
            seq += 1


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
        auto_exposure: bool = True,
        exposure: int = 140,
        gain: int = 32,
        auto_white_balance: bool = True,
        white_balance: int = 4600,
        enable_depth: bool = False,
        depth_width: int | None = None,
        depth_height: int | None = None,
        align_depth_to_color: bool = True,
        warmup_frames: int = 30,
        require_usb3: bool = False,
    ):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")
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
            cfg.enable_stream(
                rs.stream.depth,
                depth_width or width,
                depth_height or height,
                rs.format.z16,
                fps,
            )
        try:
            profile = self.pipeline.start(cfg)
            self._align = rs.align(rs.stream.color) if enable_depth and align_depth_to_color else None
            _configure_realsense_color_sensor(
                profile,
                auto_exposure=auto_exposure,
                exposure=exposure,
                gain=gain,
                auto_white_balance=auto_white_balance,
                white_balance=white_balance,
            )
            warmup_timeout_s = max(5.0, float(warmup_frames) / max(float(fps), 1.0) + 5.0)
            self._warmup(max(0, warmup_frames), timeout_s=warmup_timeout_s)
        except BaseException:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise

    def read_frameset(self) -> RealSenseFrameSet | None:
        frames = self.pipeline.poll_for_frames()
        return self._decode_frames(frames) if frames else None

    def wait_frameset(self, *, timeout_ms: int = 1000) -> RealSenseFrameSet | None:
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
            depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=False)
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
                frame = self.pipeline.wait_for_frames(1000)
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
        backend_api: str = "v4l2",
        pixel_format: str = "",
        warmup_frames: int = 10,
    ):
        source, label = resolve_video_source(device)
        self.label = label
        api = cv2.CAP_V4L2 if backend_api == "v4l2" else 0
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device={device}")
        if pixel_format:
            if len(pixel_format) != 4:
                raise ValueError(f"pixel_format must be a four-character code, got {pixel_format!r}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
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


CameraBackend = Literal["realsense", "v4l2"]
ColorCamera = RealSenseColorCamera | OpenCVColorCamera


def open_color_camera(
    backend: str,
    *,
    serial: str = "",
    device: str = "",
    width: int,
    height: int,
    fps: int,
    backend_api: str = "v4l2",
    pixel_format: str = "",
    auto_exposure: bool = True,
    exposure: int = 140,
    gain: int = 32,
    auto_white_balance: bool = True,
    white_balance: int = 4600,
    warmup_frames: int = 10,
) -> ColorCamera:
    """Open a color camera from an explicit tracked backend contract."""

    normalized = backend.strip().lower()
    if normalized == "realsense":
        if not serial:
            raise ValueError("RealSense camera backend requires an explicit serial")
        return RealSenseColorCamera(
            serial,
            width,
            height,
            fps,
            auto_exposure=auto_exposure,
            exposure=exposure,
            gain=gain,
            auto_white_balance=auto_white_balance,
            white_balance=white_balance,
            warmup_frames=warmup_frames,
        )
    if normalized == "v4l2":
        if not device:
            raise ValueError("V4L2 camera backend requires a device or 'auto'")
        return OpenCVColorCamera(
            device,
            width,
            height,
            fps,
            backend_api=backend_api,
            pixel_format=pixel_format,
            warmup_frames=warmup_frames,
        )
    raise ValueError(f"unsupported camera backend {backend!r}; expected 'realsense' or 'v4l2'")


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
        found = ", ".join(_rs_device_info(device, rs.camera_info.serial_number) for device in devices)
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
    exposure: int,
    gain: int,
    auto_white_balance: bool,
    white_balance: int,
) -> None:
    sensors = profile.get_device().query_sensors()
    color_sensor = _find_realsense_color_sensor(sensors)
    if color_sensor is None:
        return
    color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
    if not auto_exposure:
        color_sensor.set_option(rs.option.exposure, float(exposure))
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0)
    if not auto_white_balance and color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def _find_realsense_color_sensor(sensors: Any) -> Any | None:
    for sensor in sensors:
        if sensor.supports(rs.option.enable_auto_white_balance):
            return sensor
    if len(sensors) >= 2:
        return sensors[1]
    return None
