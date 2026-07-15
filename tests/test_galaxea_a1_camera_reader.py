import threading
from pathlib import Path

import pytest

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.hardware import cameras
from galaxea_a1_runtime.hardware.cameras import LatestCameraReader


def test_camera_reader_surfaces_cross_thread_failure():
    failure = ValueError("decode failed")

    def read():
        raise failure

    reader = LatestCameraReader("test", read)
    reader.start()
    reader.wait_stopped()

    assert reader.exception() is failure
    assert not reader.is_alive()


def test_camera_reader_reports_a_read_that_cannot_stop():
    release = threading.Event()

    def read():
        release.wait()
        return None

    reader = LatestCameraReader("blocked", read)
    reader.start()
    with pytest.raises(RuntimeError, match="did not stop"):
        reader.stop(timeout_s=0.01)

    release.set()
    reader.wait_stopped()


def test_configured_camera_factory_forwards_the_complete_physical_contract(
    monkeypatch,
):
    repo = Path(__file__).resolve().parents[1]
    config = load_system_config(repo / "configs/system/a1.toml", repo_root=repo)
    captured = {}
    sentinel = object()

    def fake_open(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cameras, "RealSenseColorCamera", fake_open)

    result = cameras.open_configured_camera(
        config.cameras.front,
        warmup_frames=config.cameras.warmup_frames,
        enable_depth=False,
    )

    assert result is sentinel
    assert captured == {
        "args": ("341522300456", 640, 480, 30),
        "auto_exposure": True,
        "exposure": None,
        "gain": None,
        "auto_white_balance": True,
        "white_balance": None,
        "warmup_frames": 20,
        "require_usb3": False,
        "enable_depth": False,
        "depth_width": None,
        "depth_height": None,
        "align_depth_to_color": None,
    }
