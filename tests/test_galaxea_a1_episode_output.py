import json
from dataclasses import replace
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.teleop.metadata import (
    EpisodeMetadataRequest,
    write_metadata,
)
from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.collection.episode_output import validate_staged_episode
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]


def test_staged_episode_requires_metadata_csv_and_exact_camera_counts(tmp_path):
    for directory in ("cam0", "cam1"):
        path = tmp_path / directory
        path.mkdir()
        (path / "000000.jpg").write_bytes(b"frame")
    (tmp_path / "frames.csv").write_text(
        "frame_index,cam0_relpath,cam1_relpath\n0,cam0/000000.jpg,cam1/000000.jpg\n"
    )
    (tmp_path / "metadata.json").write_text('{"frame_count": 1}')

    validate_staged_episode(tmp_path, frame_count=1, depth_enabled=False)

    (tmp_path / "cam1" / "000001.jpg").write_bytes(b"extra")
    with pytest.raises(RuntimeError):
        validate_staged_episode(tmp_path, frame_count=1, depth_enabled=False)


def test_staged_episode_rejects_zero_frames_and_undeclared_depth(tmp_path):
    for directory in ("cam0", "cam1"):
        (tmp_path / directory).mkdir()
    (tmp_path / "frames.csv").write_text("frame_index,cam0_relpath,cam1_relpath\n")
    (tmp_path / "metadata.json").write_text('{"frame_count": 0}')

    with pytest.raises(RuntimeError):
        validate_staged_episode(tmp_path, frame_count=0, depth_enabled=False)

    for directory in ("cam0", "cam1"):
        (tmp_path / directory / "000000.jpg").write_bytes(b"frame")
    (tmp_path / "frames.csv").write_text(
        "frame_index,cam0_relpath,cam1_relpath\n0,cam0/000000.jpg,cam1/000000.jpg\n"
    )
    (tmp_path / "metadata.json").write_text('{"frame_count": 1}')
    (tmp_path / "cam0_depth").mkdir()
    with pytest.raises(RuntimeError):
        validate_staged_episode(tmp_path, frame_count=1, depth_enabled=False)


def test_episode_metadata_derives_camera_contract_from_typed_config(tmp_path):
    config = load_teleop_config(REPO / "configs/teleop/a1_so100.toml", repo_root=REPO)
    front = replace(config.system.cameras.front, depth=True)
    cameras_config = replace(config.system.cameras, front=front)
    config = replace(config, system=replace(config.system, cameras=cameras_config))
    write_metadata(
        EpisodeMetadataRequest(
            episode_dir=tmp_path,
            task="pick cube",
            experiment="pick_cube",
            episode_index=0,
            frame_count=1,
            state_mode=StateMode.EEF_JOINT,
            front_crop=config.system.cameras.front.crop,
            wrist_label="realsense:test",
            config_path="configs/teleop/a1_so100.toml",
            config=config,
        )
    )

    payload = json.loads((tmp_path / "metadata.json").read_text())
    cameras = {item["name"]: item for item in payload["cameras"]}
    assert cameras["front"]["width"] == 480
    assert cameras["front"]["height"] == 480
    assert cameras["front"]["crop_xywh"] == [103, 0, 480, 480]
    assert cameras["front_depth"]["width"] == 480
    assert cameras["front_depth"]["height"] == 480
    assert payload["quality_checks"]["max_camera_pair_skew_s"] == 0.1
    assert payload["quality_checks"]["leader_gripper_source_min"] == 0.0
    assert payload["quality_checks"]["leader_gripper_source_max"] == 53.16
    assert payload["quality_checks"]["gripper_continuous_stroke_min_mm"] == 0.0
    assert payload["quality_checks"]["gripper_continuous_stroke_max_mm"] == 104.0
