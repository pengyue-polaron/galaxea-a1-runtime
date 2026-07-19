"""Raw Camera Bridge consumption for one teleop collection process."""

from __future__ import annotations

from galaxea_a1_runtime.apps.teleop.recording import wait_for_new_camera_samples
from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeReaders
from galaxea_a1_runtime.hardware.cameras import CameraReader
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


class TeleopCameraSession:
    def __init__(self, config: TeleopConfig):
        self.config = config
        self.camera_bridge: CameraBridgeReaders | None = None
        self.front_reader: CameraReader | None = None
        self.wrist_reader: CameraReader | None = None

    def start(self) -> str:
        cameras = self.config.system.cameras
        front = cameras.front
        try:
            self.camera_bridge = CameraBridgeReaders(cameras)
            self.camera_bridge.start(timeout_s=self.config.collection.ready_timeout_s)
            self.front_reader = self.camera_bridge.front
            self.wrist_reader = self.camera_bridge.wrist
            wait_for_new_camera_samples(
                self.readers,
                min_seq={"front": -1, "wrist": -1},
                timeout_s=self.config.collection.ready_timeout_s,
            )
        except BaseException:
            self.close()
            raise
        depth = "on" if front.depth else "off"
        assert self.camera_bridge is not None
        metadata = self.camera_bridge.metadata
        return (
            f"wrist={metadata.wrist_source}, "
            f"realsense_usb={metadata.front_usb_type}, "
            f"depth={depth}"
        )

    @property
    def readers(self) -> tuple[CameraReader, CameraReader]:
        if self.front_reader is None or self.wrist_reader is None:
            raise RuntimeError("teleop cameras are not started")
        return self.front_reader, self.wrist_reader

    @property
    def wrist_label(self) -> str:
        if self.camera_bridge is None:
            raise RuntimeError("wrist camera bridge is not started")
        return self.camera_bridge.metadata.wrist_source

    def close(self) -> None:
        cleanup_error: BaseException | None = None
        if self.camera_bridge is not None:
            try:
                self.camera_bridge.close()
            except BaseException as exc:  # noqa: BLE001 - clear owned references.
                cleanup_error = exc
            self.camera_bridge = None
        self.wrist_reader = None
        self.front_reader = None
        if cleanup_error is not None:
            raise cleanup_error
