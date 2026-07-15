import copy
import tomllib
from pathlib import Path

import pytest

from galaxea_a1_runtime.configuration.cameras import (
    parse_system_cameras,
    required_front_roi,
)
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]


def test_required_front_roi_returns_tracked_square_region():
    config = load_teleop_config(REPO / "configs/teleop/a1_so100.toml", repo_root=REPO)

    roi = required_front_roi(config.system.cameras)

    assert roi is not None
    assert roi.xywh == (103, 0, 480, 480)


def _camera_tables():
    raw = tomllib.loads((REPO / "configs/system/a1.toml").read_text())
    return copy.deepcopy(raw["cameras"])


@pytest.mark.parametrize(
    ("camera", "key", "value"),
    [
        ("wrist", "crop_x", 0),
        ("front", "exposure", 140),
        ("front", "white_balance", 4600),
        ("front", "depth_width", 640),
    ],
)
def test_disabled_camera_features_reject_ignored_values(camera, key, value):
    cameras = _camera_tables()
    cameras[camera][key] = value

    with pytest.raises(ValueError, match="unknown"):
        parse_system_cameras(cameras)


def test_enabled_camera_features_require_their_settings():
    cameras = _camera_tables()
    cameras["front"]["depth"] = True

    with pytest.raises(ValueError, match="missing"):
        parse_system_cameras(cameras)


def test_manual_sensor_and_depth_modes_parse_only_explicit_values():
    cameras = _camera_tables()
    front = cameras["front"]
    front.update(
        {
            "depth": True,
            "depth_width": 640,
            "depth_height": 480,
            "align_depth_to_color": True,
            "auto_exposure": False,
            "exposure": 140,
            "gain": 32,
            "auto_white_balance": False,
            "white_balance": 4600,
        }
    )

    config = parse_system_cameras(cameras)

    assert config.front.depth is True
    assert (config.front.depth_width, config.front.depth_height) == (640, 480)
    assert (config.front.exposure, config.front.gain) == (140, 32)
    assert config.front.white_balance == 4600
