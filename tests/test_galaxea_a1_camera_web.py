import http.client
from pathlib import Path
import socket
import time
import json

import numpy as np

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
from galaxea_a1_runtime.hardware.cameras import CameraSample
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    WebPreviewConfig,
    color_from_bgr,
)
from galaxea_a1_runtime.hardware.image_geometry import crop_image, draw_image_roi
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]


class FakeReader:
    def __init__(self, value: np.ndarray):
        self.sample = CameraSample(seq=1, monotonic_s=time.perf_counter(), value=value)

    def latest(self) -> CameraSample:
        return self.sample


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_camera_web_serves_health_without_authentication():
    config = WebPreviewConfig(
        enabled=True,
        bind="127.0.0.1",
        port=_free_port(),
        fps=20.0,
        jpeg_quality=70,
    )
    image = np.full((48, 64, 3), 127, dtype=np.uint8)
    preview = CameraWebPreview(config, max_source_age_s=0.5)
    roi = ImageRoi(x=8, y=0, width=48, height=48)
    preview.register_reader(
        "agent",
        FakeReader(image),
        extract=color_from_bgr,
        source="agent-test",
        overlay_roi=roi,
        overlay_label="RECORDED 48x48",
    )
    preview.register_reader(
        "wrist", FakeReader(image), extract=color_from_bgr, source="wrist-test"
    )
    preview.start()
    try:
        time.sleep(0.15)
        connection = http.client.HTTPConnection("127.0.0.1", config.port, timeout=2)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        health = json.loads(body)
        assert health["ok"] is True
        assert health["streams"]["agent"]["overlay_roi_xywh"] == [8, 0, 48, 48]

        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"Galaxea A1" in response.read()
        connection.close()
    finally:
        preview.close()


def test_camera_web_health_fails_when_sources_become_stale():
    config = WebPreviewConfig(
        enabled=True,
        bind="127.0.0.1",
        port=_free_port(),
        fps=30.0,
        jpeg_quality=70,
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    preview = CameraWebPreview(config, max_source_age_s=0.05)
    preview.register_reader(
        "agent", FakeReader(image), extract=color_from_bgr, source="agent-test"
    )
    preview.register_reader(
        "wrist", FakeReader(image), extract=color_from_bgr, source="wrist-test"
    )
    preview.start()
    try:
        time.sleep(0.12)
        connection = http.client.HTTPConnection("127.0.0.1", config.port, timeout=2)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        health = json.loads(response.read())
        assert response.status == 503
        assert health["ok"] is False
        assert health["streams"]["agent"]["fresh"] is False
        connection.close()
    finally:
        preview.close()


def test_image_roi_crops_exact_pixels_and_draws_non_destructively():
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    original = image.copy()
    roi = ImageRoi(x=2, y=1, width=6, height=6)

    cropped = crop_image(image, roi)
    overlay = draw_image_roi(image, roi, label="")

    assert cropped.shape == (6, 6, 3)
    assert np.array_equal(cropped, image[1:7, 2:8])
    assert np.array_equal(image, original)
    assert not np.array_equal(overlay, original)


def test_agentview_roi_is_identical_across_collection_and_inference_configs():
    teleop = load_teleop_config(REPO / "configs/teleop/a1_so100.toml", repo_root=REPO)
    lingbot = load_lingbot_config(
        REPO / "configs/deployments/lingbot/fruit_placement_eef.toml", repo_root=REPO
    )
    pi05 = load_pi05_config(
        REPO / "configs/deployments/pi05/fruit_placement_eef.toml", repo_root=REPO
    )

    assert teleop.system.cameras.front.crop is not None
    assert lingbot.system.cameras.front.crop is not None
    assert pi05.system.cameras.front.crop is not None
    assert {
        teleop.system.cameras.front.crop.xywh,
        lingbot.system.cameras.front.crop.xywh,
        pi05.system.cameras.front.crop.xywh,
    } == {(103, 0, 480, 480)}
