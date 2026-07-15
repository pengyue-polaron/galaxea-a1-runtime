"""Camera ownership for one teleop collection process."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.apps.teleop.recording import wait_for_new_camera_samples
from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    open_color_camera,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
    web_preview_config_from_args,
)


class TeleopCameraSession:
    def __init__(self, args: Any, front_crop: ImageRoi | None):
        self.args = args
        self.front_crop = front_crop
        self.front: RealSenseColorCamera | None = None
        self.wrist: ColorCamera | None = None
        self.front_reader: LatestCameraReader | None = None
        self.wrist_reader: LatestCameraReader | None = None
        self.preview: CameraWebPreview | None = None

    def start(self) -> str:
        try:
            self.front = RealSenseColorCamera(
                self.args.cam0_serial,
                self.args.cam0_width,
                self.args.cam0_height,
                self.args.cam0_fps,
                enable_depth=self.args.cam0_depth_enabled,
                depth_width=self.args.cam0_depth_width,
                depth_height=self.args.cam0_depth_height,
                align_depth_to_color=self.args.cam0_align_depth_to_color,
                warmup_frames=20,
                require_usb3=self.args.cam0_require_usb3,
            )
            self.wrist = open_color_camera(
                self.args.cam1_backend,
                serial=self.args.cam1_serial,
                device=self.args.cam1_device,
                width=self.args.cam1_width,
                height=self.args.cam1_height,
                fps=self.args.cam1_fps,
                pixel_format=self.args.cam1_pixel_format,
                warmup_frames=10,
            )
            self.front_reader = LatestCameraReader("front", self.front.read_frameset)
            self.wrist_reader = LatestCameraReader("wrist", self.wrist.read_bgr)
            self.front_reader.start()
            self.wrist_reader.start()
            self._start_preview()
            wait_for_new_camera_samples(
                self.readers,
                min_seq={"front": -1, "wrist": -1},
                timeout_s=self.args.ready_timeout_s,
            )
        except BaseException:
            self.close()
            raise
        depth = "on" if self.args.cam0_depth_enabled else "off"
        return (
            f"wrist={self.wrist.label}, realsense_usb={self.front.usb_type}, "
            f"depth={depth}"
        )

    @property
    def readers(self) -> tuple[LatestCameraReader, LatestCameraReader]:
        if self.front_reader is None or self.wrist_reader is None:
            raise RuntimeError("teleop cameras are not started")
        return self.front_reader, self.wrist_reader

    @property
    def wrist_label(self) -> str:
        if self.wrist is None:
            raise RuntimeError("wrist camera is not started")
        return self.wrist.label

    def close(self) -> None:
        if self.preview is not None:
            self.preview.close()
            self.preview = None
        for reader in (self.wrist_reader, self.front_reader):
            if reader is not None:
                reader.stop()
        self.wrist_reader = None
        self.front_reader = None
        for camera in (self.wrist, self.front):
            if camera is not None:
                camera.close()
        self.wrist = None
        self.front = None

    def _start_preview(self) -> None:
        preview_config = web_preview_config_from_args(self.args)
        if not preview_config.enabled:
            return
        if self.front is None or self.wrist is None:
            raise RuntimeError("cannot start preview before cameras")
        front_reader, wrist_reader = self.readers
        self.preview = CameraWebPreview(preview_config)
        self.preview.register_reader(
            "agent",
            front_reader,
            extract=color_from_frameset,
            source=self.front.label,
            overlay_roi=self.front_crop,
            overlay_label=(
                f"RECORDED {self.front_crop.width}x{self.front_crop.height}"
                if self.front_crop is not None
                else ""
            ),
        )
        self.preview.register_reader(
            "wrist",
            wrist_reader,
            extract=color_from_bgr,
            source=self.wrist.label,
        )
        self.preview.start()
