from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from galaxea_a1_runtime.configuration.cameras import (
    SystemCamerasConfig,
    SystemRealSenseCameraConfig,
)
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.hardware.camera_bridge import (
    CameraBridgeReaders,
    CameraBridgeServer,
)
from galaxea_a1_runtime.hardware.camera_reader import CameraSample
from galaxea_a1_runtime.hardware.cameras import RealSenseFrameSet


class FakeReader:
    def __init__(self, name: str, sample: CameraSample):
        self.name = name
        self._sample = sample
        self._lock = threading.Lock()

    def latest(self) -> CameraSample:
        with self._lock:
            return self._sample

    def latest_seq(self) -> int:
        return self.latest().seq

    def frame_count(self) -> int:
        return self.latest_seq() + 1

    def exception(self) -> None:
        return None


def _camera_config(*, depth: bool) -> SystemCamerasConfig:
    front = SystemRealSenseCameraConfig(
        backend="realsense",
        width=8,
        height=6,
        fps=30,
        crop=ImageRoi(x=1, y=0, width=6, height=6),
        serial="front-serial",
        require_usb3=False,
        depth=depth,
        depth_width=8 if depth else None,
        depth_height=6 if depth else None,
        align_depth_to_color=True if depth else None,
        auto_exposure=True,
        exposure=None,
        gain=None,
        auto_white_balance=True,
        white_balance=None,
    )
    wrist = SystemRealSenseCameraConfig(
        backend="realsense",
        width=8,
        height=6,
        fps=30,
        crop=None,
        serial="wrist-serial",
        require_usb3=False,
        depth=False,
        depth_width=None,
        depth_height=None,
        align_depth_to_color=None,
        auto_exposure=True,
        exposure=None,
        gain=None,
        auto_white_balance=True,
        white_balance=None,
    )
    return SystemCamerasConfig(
        warmup_frames=0,
        max_age_s=0.5,
        max_pair_skew_s=0.1,
        front=front,
        wrist=wrist,
    )


def test_camera_bridge_preserves_exact_raw_arrays_sequence_and_timestamps(
    tmp_path: Path,
):
    config = _camera_config(depth=True)
    socket_path = tmp_path / "camera.sock"
    front_color = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    front_depth = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    wrist_color = np.flip(front_color, axis=1).copy()
    front_original = front_color.copy()
    wrist_original = wrist_color.copy()
    front_reader = FakeReader(
        "front",
        CameraSample(
            seq=41,
            monotonic_s=1234.5,
            value=RealSenseFrameSet(front_color, front_depth),
        ),
    )
    wrist_reader = FakeReader(
        "wrist",
        CameraSample(seq=72, monotonic_s=1234.52, value=wrist_color),
    )
    server = CameraBridgeServer(
        config,
        front_reader=front_reader,
        wrist_reader=wrist_reader,
        front_source="front-source",
        wrist_source="wrist-source",
        front_usb_type="3.2",
        socket_path=socket_path,
    )
    client = CameraBridgeReaders(config, socket_path=socket_path)
    server.start()
    try:
        client.start(timeout_s=2.0)
        front = client.front.latest()
        wrist = client.wrist.latest()

        assert front is not None
        assert wrist is not None
        assert front.seq == 41
        assert front.monotonic_s == 1234.5
        assert wrist.seq == 72
        assert wrist.monotonic_s == 1234.52
        assert np.array_equal(front.value.color_bgr, front_original)
        assert np.array_equal(front.value.depth_mm, front_depth)
        assert np.array_equal(wrist.value, wrist_original)
        assert np.array_equal(front_color, front_original)
        assert np.array_equal(wrist_color, wrist_original)
        assert client.metadata.front_source == "front-source"
        assert client.metadata.wrist_source == "wrist-source"
        assert client.metadata.front_usb_type == "3.2"
        assert client.metadata.depth_enabled is True
    finally:
        client.close()
        server.close()
    assert not socket_path.exists()


def test_camera_bridge_rejects_a_consumer_with_a_different_camera_contract(
    tmp_path: Path,
):
    config = _camera_config(depth=False)
    socket_path = tmp_path / "camera.sock"
    now = time.perf_counter()
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    server = CameraBridgeServer(
        config,
        front_reader=FakeReader(
            "front", CameraSample(0, now, RealSenseFrameSet(image))
        ),
        wrist_reader=FakeReader("wrist", CameraSample(0, now, image)),
        front_source="front-source",
        wrist_source="wrist-source",
        front_usb_type="3.2",
        socket_path=socket_path,
    )
    client = CameraBridgeReaders(
        replace(config, max_pair_skew_s=0.05),
        socket_path=socket_path,
    )
    server.start()
    try:
        with pytest.raises(RuntimeError, match="persistent camera bridge"):
            client.start(timeout_s=2.0)
    finally:
        client.close()
        server.close()
