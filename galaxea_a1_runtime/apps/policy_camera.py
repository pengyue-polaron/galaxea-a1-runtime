"""Shared dual-camera ownership for RGB policy inference apps."""

from __future__ import annotations

import time
from typing import Any

from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    RealSenseFrameSet,
    open_color_camera,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, crop_image
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
    web_preview_config_from_args,
)


class PolicyCameraSession:
    def __init__(self, args: Any, front_roi: ImageRoi):
        self.args = args
        self.front_roi = front_roi
        self.front_camera: RealSenseColorCamera | None = None
        self.wrist_camera: ColorCamera | None = None
        self.front_reader: LatestCameraReader | None = None
        self.wrist_reader: LatestCameraReader | None = None
        self.preview: CameraWebPreview | None = None
        try:
            self.front_camera = RealSenseColorCamera(
                args.cam0_serial,
                args.cam_width,
                args.cam_height,
                args.cam_fps,
                auto_exposure=args.cam0_auto_exposure,
                exposure=args.cam0_exposure,
                gain=args.cam0_gain,
                auto_white_balance=args.cam0_auto_white_balance,
                white_balance=args.cam0_white_balance,
                warmup_frames=getattr(args, "camera_warmup_frames", 20),
            )
            self.wrist_camera = open_color_camera(
                args.cam1_backend,
                serial=args.cam1_serial,
                device=args.cam1_device,
                width=args.cam_width,
                height=args.cam_height,
                fps=args.cam_fps,
                backend_api=args.cam1_backend_api,
                pixel_format=getattr(args, "cam1_pixel_format", "YUYV"),
                warmup_frames=getattr(args, "camera_warmup_frames", 20),
            )
            self.front_reader = LatestCameraReader(
                "front", self.front_camera.read_frameset
            )
            self.wrist_reader = LatestCameraReader("wrist", self.wrist_camera.read_bgr)
            self.front_reader.start()
            self.wrist_reader.start()
            self._start_preview()
        except BaseException:
            self.close()
            raise

    def read_pair(self) -> tuple[Any, Any] | None:
        front_reader, wrist_reader = self._readers()
        for reader in (front_reader, wrist_reader):
            exc = reader.exception()
            if exc is not None:
                raise RuntimeError(f"{reader.name} camera reader failed") from exc
        front = front_reader.latest()
        wrist = wrist_reader.latest()
        now = time.perf_counter()
        if (
            front is None
            or wrist is None
            or now - front.monotonic_s > self.args.max_camera_age
            or now - wrist.monotonic_s > self.args.max_camera_age
        ):
            return None
        frameset = front.value
        if not isinstance(frameset, RealSenseFrameSet):
            raise RuntimeError("front camera did not return a RealSenseFrameSet")
        return (
            crop_image(
                frameset.color_bgr,
                self.front_roi,
                label="AgentView inference frame",
            ),
            wrist.value,
        )

    def wait_pair(self, *, timeout_s: float, is_shutdown) -> tuple[Any, Any]:
        deadline = time.monotonic() + timeout_s
        while not is_shutdown() and time.monotonic() < deadline:
            pair = self.read_pair()
            if pair is not None:
                return pair
            time.sleep(0.02)
        raise RuntimeError("No fresh camera pair within timeout")

    def read_observation(self, *, front_key: str, wrist_key: str) -> dict | None:
        pair = self.read_pair()
        if pair is None:
            return None
        front_bgr, wrist_bgr = pair
        return {
            front_key: front_bgr[..., ::-1].copy(),
            wrist_key: wrist_bgr[..., ::-1].copy(),
        }

    def close(self) -> None:
        if self.preview is not None:
            self.preview.close()
            self.preview = None
        for reader in (self.wrist_reader, self.front_reader):
            if reader is not None:
                reader.stop()
        self.wrist_reader = None
        self.front_reader = None
        for camera in (self.wrist_camera, self.front_camera):
            if camera is not None:
                camera.close()
        self.wrist_camera = None
        self.front_camera = None

    def _start_preview(self) -> None:
        preview_config = web_preview_config_from_args(self.args)
        if not preview_config.enabled:
            return
        if self.front_camera is None or self.wrist_camera is None:
            raise RuntimeError("cannot start preview before cameras")
        front_reader, wrist_reader = self._readers()
        self.preview = CameraWebPreview(preview_config)
        self.preview.register_reader(
            "agent",
            front_reader,
            extract=color_from_frameset,
            source=self.front_camera.label,
            overlay_roi=self.front_roi,
            overlay_label=(
                f"POLICY INPUT {self.front_roi.width}x{self.front_roi.height}"
            ),
        )
        self.preview.register_reader(
            "wrist",
            wrist_reader,
            extract=color_from_bgr,
            source=self.wrist_camera.label,
        )
        self.preview.start()

    def _readers(self) -> tuple[LatestCameraReader, LatestCameraReader]:
        if self.front_reader is None or self.wrist_reader is None:
            raise RuntimeError("policy cameras are not started")
        return self.front_reader, self.wrist_reader


def required_square_front_roi(args: Any) -> ImageRoi:
    if not args.cam0_crop_enabled:
        raise ValueError(
            "--cam0-crop-enabled is required for the inference input contract"
        )
    roi = ImageRoi(
        x=args.cam0_crop_x,
        y=args.cam0_crop_y,
        width=args.cam0_crop_width,
        height=args.cam0_crop_height,
    )
    roi.validate(
        image_width=args.cam_width,
        image_height=args.cam_height,
        label="AgentView inference crop",
    )
    if roi.width != roi.height:
        raise ValueError(
            f"AgentView inference crop must be square, got {roi.width}x{roi.height}"
        )
    return roi
