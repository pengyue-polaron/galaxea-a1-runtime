import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from galaxea_a1_runtime.lerobot.convert_raw import (
    discover_raw_dataset,
    iter_episode_frames,
    legacy_joint_contract,
    raw_episode_contract,
)


def make_raw_episode(root: Path) -> None:
    source = root / "raw_task"
    episode = source / "episode_000_20260708_120000"
    (episode / "cam0").mkdir(parents=True)
    (episode / "cam1").mkdir(parents=True)
    (source / "task.txt").write_text("pick cube\n")
    (episode / "metadata.json").write_text(
        json.dumps({"fps_target": 20, "joint_names": ["joint_1", "joint_2", "gripper"]})
    )
    pd.DataFrame(
        {
            "frame_index": [0, 1],
            "ros_stamp_s": [10.0, 10.05],
            "cam0_relpath": ["cam0/000000.jpg", "cam0/000001.jpg"],
            "cam1_relpath": ["cam1/000000.jpg", "cam1/000001.jpg"],
            "joint_1": [0.1, 0.2],
            "joint_2": [0.3, 0.4],
            "gripper": [0.0, 1.0],
        }
    ).to_csv(episode / "frames.csv", index=False)
    for camera in ("cam0", "cam1"):
        for index in (0, 1):
            image = Image.fromarray(np.full((4, 5, 3), index, dtype=np.uint8))
            image.save(episode / camera / f"{index:06d}.jpg")


def test_discover_raw_dataset(tmp_path):
    make_raw_episode(tmp_path)

    summary = discover_raw_dataset(source_root=tmp_path / "raw_task")

    assert summary.task == "pick cube"
    assert summary.total_frames == 2
    assert summary.episodes[0].joint_names == ("joint_1", "joint_2", "gripper")
    assert [camera.name for camera in summary.episodes[0].camera_specs] == ["front", "wrist"]


def test_iter_episode_frames_uses_next_state_as_action(tmp_path):
    make_raw_episode(tmp_path)
    summary = discover_raw_dataset(source_root=tmp_path / "raw_task")
    episode = summary.episodes[0]
    contract = legacy_joint_contract(
        joint_names=episode.joint_names,
        camera_specs=episode.camera_specs,
    )

    frames = list(iter_episode_frames(episode=episode, task=summary.task, contract=contract))

    assert frames[0]["observation.state"] == pytest.approx((0.1, 0.3, 0.0))
    assert frames[0]["action"] == pytest.approx((0.2, 0.4, 1.0))
    assert frames[0]["observation.state"].dtype == np.float32
    assert frames[0]["action"].dtype == np.float32
    assert "timestamp" not in frames[0]
    assert frames[1]["action"] == pytest.approx((0.2, 0.4, 1.0))
    assert frames[0]["observation.images.front"].shape == (4, 5, 3)


def make_teleop_raw_episode(root: Path) -> None:
    source = root / "teleop_task"
    episode = source / "episode_000_20260708_120000"
    (episode / "cam0").mkdir(parents=True)
    (episode / "cam1").mkdir(parents=True)
    (episode / "cam0_depth").mkdir(parents=True)
    (source / "task.txt").write_text("pick cube\n")
    state_names = [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_qx",
        "eef_qy",
        "eef_qz",
        "eef_qw",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]
    action_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "galaxea_a1_teleop_raw_v2",
                "fps_target": 30,
                "state_names": state_names,
                "action_names": action_names,
            }
        )
    )
    row = {
        "frame_index": 0,
        "ros_stamp_s": 10.0,
        "cam0_relpath": "cam0/000000.jpg",
        "cam1_relpath": "cam1/000000.jpg",
        "cam0_depth_relpath": "cam0_depth/000000.png",
    }
    row.update({f"state.{name}": float(index) for index, name in enumerate(state_names)})
    row.update({f"action.{name}": float(index + 10) for index, name in enumerate(action_names)})
    pd.DataFrame([row]).to_csv(episode / "frames.csv", index=False)
    for camera in ("cam0", "cam1"):
        image = Image.fromarray(np.full((4, 5, 3), 7, dtype=np.uint8))
        image.save(episode / camera / "000000.jpg")
    depth = Image.fromarray(np.full((4, 5), 1234, dtype=np.uint16))
    depth.save(episode / "cam0_depth" / "000000.png")


def test_iter_episode_frames_preserves_new_teleop_state_and_action(tmp_path):
    make_teleop_raw_episode(tmp_path)
    summary = discover_raw_dataset(source_root=tmp_path / "teleop_task")
    episode = summary.episodes[0]
    contract = raw_episode_contract(
        state_names=episode.state_names,
        action_names=episode.action_names,
        camera_specs=episode.camera_specs,
    )

    frames = list(iter_episode_frames(episode=episode, task=summary.task, contract=contract))

    assert episode.schema_version == "galaxea_a1_teleop_raw_v2"
    assert frames[0]["observation.state"] == pytest.approx(tuple(float(i) for i in range(14)))
    assert frames[0]["action"] == pytest.approx(tuple(float(i + 10) for i in range(7)))
    assert frames[0]["observation.state"].dtype == np.float32
    assert frames[0]["action"].dtype == np.float32
    assert "timestamp" not in frames[0]
    assert [camera.name for camera in episode.camera_specs] == ["front", "wrist", "front_depth"]
    assert episode.camera_specs[2].is_depth_map is True
    assert frames[0]["observation.images.front_depth"].shape == (4, 5, 1)
    assert frames[0]["observation.images.front_depth"].dtype == np.uint16
