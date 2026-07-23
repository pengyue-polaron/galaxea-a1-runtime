"""Shared raw Camera Bridge consumption for RGB policy inference apps."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeReaders
from galaxea_a1_runtime.hardware.cameras import CameraReader, RealSenseFrameSet
from galaxea_a1_runtime.hardware.image_geometry import crop_image
from galaxea_a1_runtime.hardware.web_preview import color_from_frameset
from galaxea_a1_runtime.hardware.video_recorder import (
    PairedCameraVideoRecorder,
    PairedVideoRecordingResult,
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
        self.front_roi = required_front_roi(system.cameras)
        self.camera_bridge: CameraBridgeReaders | None = None
        self.front_reader: CameraReader | None = None
        self.wrist_reader: CameraReader | None = None
        self.camera_recorder: PairedCameraVideoRecorder | None = None
        self.recording_result: PairedVideoRecordingResult | None = None
        try:
            self.camera_bridge = CameraBridgeReaders(system.cameras)
            self.camera_bridge.start(timeout_s=system.web_preview.startup_timeout_s)
            self.front_reader = self.camera_bridge.front
            self.wrist_reader = self.camera_bridge.wrist
        except BaseException:
            self.close()
            raise

    def read_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
        front_reader, wrist_reader = self._readers()
        if self.camera_recorder is not None:
            recording_error = self.camera_recorder.exception()
            if recording_error is not None:
                raise RuntimeError(
                    "paired camera video recorder failed"
                ) from recording_error
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
        errors: list[BaseException] = []
        if self.camera_recorder is not None:
            try:
                self.recording_result = self.camera_recorder.close()
            except BaseException as exc:  # Finalize other camera resources too.
                errors.append(exc)
            self.camera_recorder = None
        if self.camera_bridge is not None:
            try:
                self.camera_bridge.close()
            except BaseException as exc:  # Finalize all owned resources.
                errors.append(exc)
            self.camera_bridge = None
        self.wrist_reader = None
        self.front_reader = None
        if errors:
            raise BaseExceptionGroup("policy camera cleanup failed", errors)

    def start_camera_recording(
        self,
        *,
        output_root: Path,
        run_id: str,
        front_video_filename: str = "front.mp4",
        wrist_video_filename: str = "wrist.mp4",
    ) -> tuple[Path, Path]:
        if self.camera_recorder is not None:
            raise RuntimeError("paired camera recording is already active")
        self._readers()
        front = self.system.cameras.front
        wrist = self.system.cameras.wrist
        if front.fps != wrist.fps:
            raise ValueError(
                "paired camera recording requires matching front/wrist fps, "
                f"got {front.fps} and {wrist.fps}"
            )
        if self.camera_bridge is None:
            raise RuntimeError("cannot record cameras before their bridge is open")
        self.camera_recorder = PairedCameraVideoRecorder(
            read_pair=self.camera_bridge.latest_pair,
            reader_exception=self.camera_bridge.exception,
            extract_front_bgr=color_from_frameset,
            extract_wrist_bgr=lambda value: value,
            output_root=output_root,
            run_id=run_id,
            front_width=front.width,
            front_height=front.height,
            wrist_width=wrist.width,
            wrist_height=wrist.height,
            fps=front.fps,
            front_source=self.camera_bridge.metadata.front_source,
            wrist_source=self.camera_bridge.metadata.wrist_source,
            max_source_age_s=self.system.cameras.max_age_s,
            max_pair_skew_s=self.system.cameras.max_pair_skew_s,
            front_video_filename=front_video_filename,
            wrist_video_filename=wrist_video_filename,
        )
        try:
            self.camera_recorder.start()
        except BaseException:
            self.camera_recorder = None
            raise
        return (
            self.camera_recorder.front_path,
            self.camera_recorder.wrist_path,
        )

    def recording_progress(self) -> tuple[int, float] | None:
        if self.camera_recorder is None:
            return None
        return self.camera_recorder.frame_count, self.camera_recorder.elapsed_s

    def _readers(self) -> tuple[CameraReader, CameraReader]:
        if self.front_reader is None or self.wrist_reader is None:
            raise RuntimeError("policy cameras are not started")
        return self.front_reader, self.wrist_reader
