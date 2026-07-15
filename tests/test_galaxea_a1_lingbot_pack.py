from pathlib import Path

import numpy as np
import pytest

import galaxea_a1_runtime.lerobot.lingbot_pack as lingbot_pack_module
from galaxea_a1_runtime.kinematics import (
    SerialChainFK,
    compose_relative_pose,
    relative_pose,
)
from galaxea_a1_runtime.lerobot.lingbot_pack import (
    ACTION_NAMES,
)
from galaxea_a1_runtime.lerobot.lingbot_pack_config import load_pack_config
from galaxea_a1_runtime.lerobot.dataset_package import json_value
from galaxea_a1_runtime.schema import (
    JOINT_ACTION_NAMES_RAD,
    LINGBOT_EEF_ACTION_CHANNEL_IDS,
)

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
    assert LINGBOT_EEF_ACTION_CHANNEL_IDS == (0, 1, 2, 3, 4, 5, 6, 28)
    assert JOINT_ACTION_NAMES_RAD == (
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
    assert config.raw_source_root.name == "banana_in_the_plate"
    assert config.source_root.name == "banana_in_the_plate_lerobot_v3"
    assert config.source_repo_id == "galaxea-a1/banana_in_the_plate_lerobot_v3"
    assert config.overwrite is True
    assert config.source_contract.state_names[-1] == "gripper"
    assert len(config.source_contract.state_names) == 14
    assert (
        config.source_contract.camera_specs[0].height,
        config.source_contract.camera_specs[0].width,
    ) == (480, 480)
    assert config.v3_target_root.name == "banana_in_the_plate_lingbot_eef_continuous_v3"
    assert (
        config.v21_target_root.name == "banana_in_the_plate_lingbot_eef_continuous_v21"
    )
    assert config.joint_v3_target_root.name == "banana_in_the_plate_joint_continuous_v3"
    assert config.urdf_path == URDF
    assert config.gripper_stroke_min_mm == 0.0
    assert config.gripper_stroke_max_mm == 104.0


def test_lingbot_pack_config_rejects_unknown_keys(tmp_path):
    source = REPO_ROOT / "configs/datasets/banana_in_the_plate.toml"
    path = tmp_path / "dataset.toml"
    path.write_text(source.read_text() + "\nunexpected = true\n")

    with pytest.raises(ValueError):
        load_pack_config(path)


def test_v21_json_conversion_handles_nested_numpy_values():
    value = np.array([np.array([1.0]), np.array([2.0])], dtype=object)
    assert json_value(value) == [[1.0], [2.0]]


def test_dataset_command_runs_raw_conversion_before_all_packages(monkeypatch):
    calls = []

    def record(name, result=None):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return {} if result is None else result

        return invoke

    monkeypatch.setattr(
        lingbot_pack_module, "convert_raw_dataset", record("raw_to_lerobot")
    )
    monkeypatch.setattr(
        lingbot_pack_module, "pack_lingbot_dataset", record("lingbot_v3")
    )
    monkeypatch.setattr(
        lingbot_pack_module, "export_v21_dataset", record("lingbot_v21")
    )
    monkeypatch.setattr(
        lingbot_pack_module, "pack_joint_v3_dataset", record("joint_v3")
    )

    result = lingbot_pack_module.main(
        ["--config", str(REPO_ROOT / "configs/datasets/banana_in_the_plate.toml")]
    )

    assert result == 0
    assert [name for name, _ in calls] == [
        "raw_to_lerobot",
        "lingbot_v3",
        "lingbot_v21",
        "joint_v3",
    ]
    assert calls[0][1]["source_root"] == REPO_ROOT / "data/raw/banana_in_the_plate"
    assert calls[0][1]["overwrite"] is True
