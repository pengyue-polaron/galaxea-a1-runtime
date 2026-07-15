from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import galaxea_a1_runtime.lerobot.pipeline as pipeline_module
import galaxea_a1_runtime.lerobot.pipeline_config as pipeline_config_module
from galaxea_a1_runtime.configuration.base import load_toml
from galaxea_a1_runtime.kinematics import (
    SerialChainFK,
    compose_relative_pose,
    relative_pose,
)
from galaxea_a1_runtime.lerobot.dataset_package import json_value
from galaxea_a1_runtime.lerobot.eef_pack import EEF_ACTION_NAMES
from galaxea_a1_runtime.lerobot.pipeline_config import (
    load_pipeline_config,
)
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES_RAD

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG_FIXTURE = REPO_ROOT / "tests/fixtures/dataset_pipeline.toml"
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


def test_generic_a1_action_contracts():
    assert EEF_ACTION_NAMES == (
        "eef_delta_x_from_episode_start",
        "eef_delta_y_from_episode_start",
        "eef_delta_z_from_episode_start",
        "eef_delta_qx_from_episode_start",
        "eef_delta_qy_from_episode_start",
        "eef_delta_qz_from_episode_start",
        "eef_delta_qw_from_episode_start",
        "gripper_normalized",
    )
    assert JOINT_ACTION_NAMES_RAD == (
        "joint_1_rad",
        "joint_2_rad",
        "joint_3_rad",
        "joint_4_rad",
        "joint_5_rad",
        "joint_6_rad",
        "gripper_normalized",
    )


def test_dataset_pipeline_config_fixture():
    config = load_pipeline_config(PIPELINE_CONFIG_FIXTURE)
    assert config.raw_source_root.name == "test_experiment"
    assert config.joint_v3_target_root.name == "test_experiment_joint_v3"
    assert config.joint_v3_repo_id == "galaxea-a1/test_experiment_joint_v3"
    assert config.joint_v21_target_root.name == "test_experiment_joint_v21"
    assert config.joint_v21_repo_id == "galaxea-a1/test_experiment_joint_v21"
    assert config.overwrite is True
    assert config.source_contract.state_names[-1] == "gripper"
    assert len(config.source_contract.state_names) == 14
    assert (
        config.source_contract.camera_specs[0].height,
        config.source_contract.camera_specs[0].width,
    ) == (480, 480)
    assert config.eef_v3_target_root.name == "test_experiment_eef_v3"
    assert config.eef_v21_target_root.name == "test_experiment_eef_v21"
    assert config.urdf_path == URDF
    assert config.gripper_stroke_min_mm == 0.0
    assert config.gripper_stroke_max_mm == 104.0


def test_dataset_pipeline_config_rejects_unknown_keys(tmp_path, monkeypatch):
    path = tmp_path / "dataset.toml"
    path.write_text(PIPELINE_CONFIG_FIXTURE.read_text() + "\nunexpected = true\n")
    monkeypatch.setattr(
        pipeline_config_module,
        "load_toml",
        lambda config_path: load_toml(config_path, repo_root=REPO_ROOT),
    )

    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        load_pipeline_config(path)


def test_v21_json_conversion_handles_nested_numpy_values():
    value = np.array([np.array([1.0]), np.array([2.0])], dtype=object)
    assert json_value(value) == [[1.0], [2.0]]


def test_dataset_command_builds_each_output_from_raw_v3(tmp_path, monkeypatch):
    calls = []

    def record(name, result=None):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return {} if result is None else result

        return invoke

    monkeypatch.setattr(
        pipeline_module, "convert_raw_dataset", record("raw_to_lerobot")
    )
    monkeypatch.setattr(pipeline_module, "pack_joint_v3_dataset", record("joint_v3"))
    monkeypatch.setattr(pipeline_module, "pack_eef_v3_dataset", record("eef_v3"))
    monkeypatch.setattr(pipeline_module, "export_v21_dataset", record("v21"))

    config = load_pipeline_config(PIPELINE_CONFIG_FIXTURE)
    config = replace(
        config,
        joint_v3_target_root=tmp_path / "joint_v3",
        joint_v3_archive_path=tmp_path / "joint_v3.tar.gz",
        joint_v21_target_root=tmp_path / "joint_v21",
        joint_v21_archive_path=tmp_path / "joint_v21.tar.gz",
        eef_v3_target_root=tmp_path / "eef_v3",
        eef_v3_archive_path=tmp_path / "eef_v3.tar.gz",
        eef_v21_target_root=tmp_path / "eef_v21",
        eef_v21_archive_path=tmp_path / "eef_v21.tar.gz",
    )
    monkeypatch.setattr(pipeline_module, "load_pipeline_config", lambda _path: config)

    result = pipeline_module.main(["--config", str(PIPELINE_CONFIG_FIXTURE)])

    assert result == 0
    assert [name for name, _ in calls] == [
        "raw_to_lerobot",
        "joint_v3",
        "joint_v3",
        "v21",
        "eef_v3",
        "eef_v3",
        "v21",
    ]
    raw_root = REPO_ROOT / "data/raw/test_experiment"
    raw_intermediate = calls[0][1]["target_root"]
    final_roots = {
        tmp_path / "joint_v3",
        tmp_path / "joint_v21",
        tmp_path / "eef_v3",
        tmp_path / "eef_v21",
    }
    assert calls[0][1]["source_root"] == raw_root
    assert calls[0][1]["overwrite"] is False
    assert calls[1][1]["target_root"] == tmp_path / "joint_v3"
    for name, kwargs in calls[1:]:
        assert kwargs["source_dataset"] == str(raw_root)
        if name in {"joint_v3", "eef_v3"}:
            assert kwargs["source_root"] == raw_intermediate
        else:
            assert kwargs["source_root"] not in final_roots
    assert calls[4][1]["target_root"] == tmp_path / "eef_v3"


def test_dataset_command_can_build_one_selected_output(tmp_path, monkeypatch):
    calls = []

    def record(name):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return {"built": name}

        return invoke

    monkeypatch.setattr(
        pipeline_module, "convert_raw_dataset", record("raw_to_lerobot")
    )
    monkeypatch.setattr(pipeline_module, "pack_joint_v3_dataset", record("joint_v3"))
    monkeypatch.setattr(pipeline_module, "pack_eef_v3_dataset", record("eef_v3"))
    monkeypatch.setattr(pipeline_module, "export_v21_dataset", record("v21"))

    config = load_pipeline_config(PIPELINE_CONFIG_FIXTURE)
    config = replace(
        config,
        joint_v3_target_root=tmp_path / "joint_v3",
        joint_v3_archive_path=tmp_path / "joint_v3.tar.gz",
    )

    result = pipeline_module.build_datasets(config, targets=["joint-v3"])

    assert result == {"joint-v3": {"built": "joint_v3"}}
    assert [name for name, _ in calls] == ["raw_to_lerobot", "joint_v3"]
