from pathlib import Path
from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.apps.teleop.collector_setup import validate_collector_args
from galaxea_a1_runtime.collection import StateMode


def _args(**overrides):
    values = {
        "fps": 30.0,
        "max_camera_age_s": 0.5,
        "max_joint_feedback_age_s": 0.5,
        "max_eef_feedback_age_s": 0.5,
        "max_action_age_s": 0.5,
        "max_gripper_age_s": 0.5,
        "max_joint_action_step_rad": 0.35,
        "gripper_stroke_min": 0.0,
        "gripper_stroke_max": 100.0,
        "teleop_config": Path("configs/teleop/a1_so100.toml"),
        "auto_reset_after_save": True,
        "reset_runtime_script": Path("scripts/apps/teleop/a1_teleop_runtime.sh"),
        "cam0_depth_enabled": False,
        "cam0_depth_width": 640,
        "cam0_depth_height": 480,
        "state_mode": "eef_joint",
        "cam0_crop_enabled": True,
        "cam0_crop_x": 103,
        "cam0_crop_y": 0,
        "cam0_crop_width": 480,
        "cam0_crop_height": 480,
        "cam0_width": 640,
        "cam0_height": 480,
        "cam0_align_depth_to_color": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_collector_setup_returns_tracked_square_roi():
    state_mode, roi = validate_collector_args(_args())

    assert state_mode is StateMode.EEF_JOINT
    assert roi is not None
    assert roi.xywh == (103, 0, 480, 480)


def test_collector_setup_requires_config_metadata_source():
    with pytest.raises(ValueError):
        validate_collector_args(_args(teleop_config=None, auto_reset_after_save=False))
