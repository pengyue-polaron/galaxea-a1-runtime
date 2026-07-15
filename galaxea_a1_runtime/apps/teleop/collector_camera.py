"""Camera ownership for one teleop collection process."""

from __future__ import annotations

from galaxea_a1_runtime.apps.teleop.recording import wait_for_new_camera_samples
from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    open_configured_camera,
    close_camera_resources,
)
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


class TeleopCameraSession:
    def __init__(self, config: TeleopConfig, front_crop: ImageRoi | None):
        self.config = config
        self.front_crop = front_crop
        self.front: RealSenseColorCamera | None = None
        self.wrist: ColorCamera | None = None
        self.front_reader: LatestCameraReader | None = None
        self.wrist_reader: LatestCameraReader | None = None
        self.preview: CameraWebPreview | None = None

    def start(self) -> str:
        cameras = self.config.system.cameras
        front = cameras.front
        wrist = cameras.wrist
        try:
            opened_front = open_configured_camera(
                front,
                warmup_frames=cameras.warmup_frames,
                enable_depth=front.depth,
            )
            if not isinstance(opened_front, RealSenseColorCamera):
                raise RuntimeError("teleop AgentView must open as a RealSense camera")
            self.front = opened_front
            self.wrist = open_configured_camera(
                wrist,
                warmup_frames=cameras.warmup_frames,
                enable_depth=False,
            )
            self.front_reader = LatestCameraReader("front", self.front.read_frameset)
            self.wrist_reader = LatestCameraReader("wrist", self.wrist.read_bgr)
            self.front_reader.start()
            self.wrist_reader.start()
            self._start_preview()
            wait_for_new_camera_samples(
                self.readers,
                min_seq={"front": -1, "wrist": -1},
                timeout_s=self.config.collection.ready_timeout_s,
            )
        except BaseException:
            self.close()
            raise
        depth = "on" if front.depth else "off"
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
        preview_error: BaseException | None = None
        if self.preview is not None:
            try:
                self.preview.close()
            except BaseException as exc:  # noqa: BLE001 - finish camera cleanup.
                preview_error = exc
            self.preview = None
        cleanup_error: BaseException | None = None
        try:
            close_camera_resources(
                (self.wrist_reader, self.front_reader),
                (self.wrist, self.front),
            )
        except BaseException as exc:  # noqa: BLE001 - clear owned references.
            cleanup_error = exc
        self.wrist_reader = None
        self.front_reader = None
        self.wrist = None
        self.front = None
        if preview_error is not None:
            raise RuntimeError("camera web preview cleanup failed") from preview_error
        if cleanup_error is not None:
            raise cleanup_error

    def _start_preview(self) -> None:
        preview_config = self.config.system.web_preview
        if not preview_config.enabled:
            return
        if self.front is None or self.wrist is None:
            raise RuntimeError("cannot start preview before cameras")
        front_reader, wrist_reader = self.readers
        self.preview = CameraWebPreview(
            preview_config,
            max_source_age_s=self.config.system.cameras.max_age_s,
        )
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
