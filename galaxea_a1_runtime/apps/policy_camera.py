"""Shared dual-camera ownership for RGB policy inference apps."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    RealSenseFrameSet,
    open_configured_camera,
    close_camera_resources,
)
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.hardware.image_geometry import crop_image
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
)


class PolicyCameraSession:
    def __init__(
        self,
        system: SystemConfig,
        *,
        front_key: str,
        wrist_key: str,
    ):
        self.system = system
        self.front_key = front_key
        self.wrist_key = wrist_key
        front = system.cameras.front
        wrist = system.cameras.wrist
        self.front_roi = required_front_roi(system.cameras)
        self.front_camera: RealSenseColorCamera | None = None
        self.wrist_camera: ColorCamera | None = None
        self.front_reader: LatestCameraReader | None = None
        self.wrist_reader: LatestCameraReader | None = None
        self.preview: CameraWebPreview | None = None
        try:
            opened_front = open_configured_camera(
                front,
                warmup_frames=system.cameras.warmup_frames,
                enable_depth=False,
            )
            if not isinstance(opened_front, RealSenseColorCamera):
                raise RuntimeError("policy AgentView must open as a RealSense camera")
            self.front_camera = opened_front
            self.wrist_camera = open_configured_camera(
                wrist,
                warmup_frames=system.cameras.warmup_frames,
                enable_depth=False,
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

    def read_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
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
            or now - front.monotonic_s > self.system.cameras.max_age_s
            or now - wrist.monotonic_s > self.system.cameras.max_age_s
        ):
            return None
        skew_s = abs(front.monotonic_s - wrist.monotonic_s)
        if skew_s > self.system.cameras.max_pair_skew_s:
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

    def wait_pair(
        self,
        *,
        timeout_s: float,
        is_shutdown: Callable[[], bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + timeout_s
        while not is_shutdown() and time.monotonic() < deadline:
            pair = self.read_pair()
            if pair is not None:
                return pair
            time.sleep(0.02)
        raise RuntimeError("No fresh camera pair within timeout")

    def read_observation(self) -> dict[str, np.ndarray] | None:
        pair = self.read_pair()
        if pair is None:
            return None
        front_bgr, wrist_bgr = pair
        return {
            self.front_key: front_bgr[..., ::-1].copy(),
            self.wrist_key: wrist_bgr[..., ::-1].copy(),
        }

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
                (self.wrist_camera, self.front_camera),
            )
        except BaseException as exc:  # noqa: BLE001 - clear owned references.
            cleanup_error = exc
        self.wrist_reader = None
        self.front_reader = None
        self.wrist_camera = None
        self.front_camera = None
        if preview_error is not None:
            raise RuntimeError("camera web preview cleanup failed") from preview_error
        if cleanup_error is not None:
            raise cleanup_error

    def _start_preview(self) -> None:
        preview_config = self.system.web_preview
        if not preview_config.enabled:
            return
        if self.front_camera is None or self.wrist_camera is None:
            raise RuntimeError("cannot start preview before cameras")
        front_reader, wrist_reader = self._readers()
        self.preview = CameraWebPreview(
            preview_config,
            max_source_age_s=self.system.cameras.max_age_s,
        )
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
