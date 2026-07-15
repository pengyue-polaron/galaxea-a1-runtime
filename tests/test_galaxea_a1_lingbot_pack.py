from pathlib import Path

import numpy as np

from galaxea_a1_runtime.kinematics import (
    SerialChainFK,
    compose_relative_pose,
    relative_pose,
)
from galaxea_a1_runtime.lerobot.lingbot_pack import (
    ACTION_NAMES,
    USED_ACTION_CHANNEL_IDS,
    load_pack_config,
)
from galaxea_a1_runtime.lerobot.joint_pack import JOINT_ACTION_NAMES
from galaxea_a1_runtime.lerobot.dataset_package import json_value

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    REPO_ROOT
    / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
)


def test_a1_urdf_chain_and_known_reset_pose():
    chain = SerialChainFK.from_urdf(URDF, base_link="base_link", tip_link="arm_seg6")
    assert chain.joint_names == tuple(f"arm_joint{index}" for index in range(1, 7))
    pose = chain.pose(
        [0.012999535, 0.0010004044, -0.08199978, -1.5410004, 0.3540001, -0.048999786]
    )
    expected = np.array(
        [
            0.06835637,
            0.0022991674,
            0.22050981,
            -0.016684923,
            0.7965297,
            0.0016861054,
            0.6043668,
        ]
    )
    if np.dot(pose[3:], expected[3:]) < 0:
        pose[3:] *= -1
    np.testing.assert_allclose(pose, expected, atol=1e-6)


def test_relative_pose_round_trip():
    initial = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    target = np.array([0.2, 0.1, 0.4, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    reconstructed = compose_relative_pose(relative_pose(target, initial), initial)
    assert np.linalg.norm(reconstructed[:3] - target[:3]) < 1e-12
    assert abs(np.dot(reconstructed[3:], target[3:])) > 1.0 - 1e-12


def test_lingbot_a1_action_contract():
    assert len(ACTION_NAMES) == 8
    assert USED_ACTION_CHANNEL_IDS == (0, 1, 2, 3, 4, 5, 6, 28)
    assert JOINT_ACTION_NAMES == (
        "joint_1_rad",
        "joint_2_rad",
        "joint_3_rad",
        "joint_4_rad",
        "joint_5_rad",
        "joint_6_rad",
        "gripper_normalized",
    )


def test_tracked_lingbot_pack_config():
    config = load_pack_config(REPO_ROOT / "configs/datasets/banana_in_the_plate.toml")
    assert config.source_root.name == "banana_in_the_plate_lerobot_v3"
    assert config.v3_target_root.name == "banana_in_the_plate_lingbot_eef_continuous_v3"
    assert (
        config.v21_target_root.name == "banana_in_the_plate_lingbot_eef_continuous_v21"
    )
    assert config.joint_v3_target_root.name == "banana_in_the_plate_joint_continuous_v3"
    assert config.urdf_path == URDF
    assert config.gripper_stroke_min_mm == 0.0
    assert config.gripper_stroke_max_mm == 100.0


def test_v21_json_conversion_handles_nested_numpy_values():
    value = np.array([np.array([1.0]), np.array([2.0])], dtype=object)
    assert json_value(value) == [[1.0], [2.0]]
