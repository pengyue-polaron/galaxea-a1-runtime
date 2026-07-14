"""LingBot camera ownership, freshness, crop, and read-only preview."""

from __future__ import annotations

import time

from galaxea_a1_runtime.hardware.cameras import (
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


class LingBotCameraSession:
    def __init__(self, args, front_roi: ImageRoi):
        self.args = args
        self.front_roi = front_roi
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
        )
        self.wrist_camera = open_color_camera(
            args.cam1_backend,
            serial=args.cam1_serial,
            device=args.cam1_device,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.cam_fps,
            backend_api=args.cam1_backend_api,
        )
        self.front_reader = LatestCameraReader("front", self.front_camera.read_frameset)
        self.wrist_reader = LatestCameraReader("wrist", self.wrist_camera.read_bgr)
        self.front_reader.start()
        self.wrist_reader.start()
        self.preview: CameraWebPreview | None = None
        preview_config = web_preview_config_from_args(args)
        if preview_config.enabled:
            self.preview = CameraWebPreview(preview_config)
            self.preview.register_reader(
                "agent",
                self.front_reader,
                extract=color_from_frameset,
                source=self.front_camera.label,
                overlay_roi=front_roi,
                overlay_label=f"POLICY INPUT {front_roi.width}x{front_roi.height}",
            )
            self.preview.register_reader(
                "wrist",
                self.wrist_reader,
                extract=color_from_bgr,
                source=self.wrist_camera.label,
            )
            self.preview.start()

    def read_observation(self) -> dict | None:
        for reader in (self.front_reader, self.wrist_reader):
            exc = reader.exception()
            if exc is not None:
                raise RuntimeError(f"{reader.name} camera reader failed") from exc
        front_sample = self.front_reader.latest()
        wrist_sample = self.wrist_reader.latest()
        now = time.perf_counter()
        if (
            front_sample is None
            or wrist_sample is None
            or now - front_sample.monotonic_s > self.args.max_camera_age
            or now - wrist_sample.monotonic_s > self.args.max_camera_age
        ):
            return None
        frameset = front_sample.value
        if not isinstance(frameset, RealSenseFrameSet):
            raise RuntimeError("agent camera did not return a RealSenseFrameSet")
        front_bgr = crop_image(
            frameset.color_bgr,
            self.front_roi,
            label="AgentView inference frame",
        )
        return {
            self.args.cam0_observation_key: front_bgr[..., ::-1].copy(),
            self.args.cam1_observation_key: wrist_sample.value[..., ::-1].copy(),
        }

    def close(self) -> None:
        if self.preview is not None:
            self.preview.close()
        self.wrist_reader.stop()
        self.front_reader.stop()
        self.front_camera.close()
        self.wrist_camera.close()
