import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import galaxea_a1_runtime.lerobot.convert_raw as convert_raw_module
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.lerobot.convert_raw import (
    convert_raw_dataset,
    discover_raw_dataset,
    iter_episode_frames,
    raw_episode_contract,
)
from galaxea_a1_runtime.schema import DEFAULT_STATE_NAMES, JOINT_ACTION_NAMES


def make_raw_episode(
    root: Path,
    *,
    episode_index: int = 0,
    width: int = 5,
    height: int = 4,
    depth: bool = False,
) -> Path:
    source = root / "raw_task"
    episode = source / f"episode_{episode_index:03d}_20260708_120000"
    for directory in ("cam0", "cam1", *(("cam0_depth",) if depth else ())):
        (episode / directory).mkdir(parents=True)
    (source / "task.txt").write_text("pick cube\n")

    cameras = [
        {"name": "front", "directory": "cam0", "width": width, "height": height},
        {"name": "wrist", "directory": "cam1", "width": width, "height": height},
    ]
    if depth:
        cameras.append(
            {
                "name": "front_depth",
                "directory": "cam0_depth",
                "width": width,
                "height": height,
            }
        )
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": TELEOP_RAW_SCHEMA_VERSION,
                "task": "pick cube",
                "action_mode": "joint_absolute",
                "frame_count": 2,
                "fps_target": 30.0,
                "state_names": list(DEFAULT_STATE_NAMES),
                "action_names": list(JOINT_ACTION_NAMES),
                "cameras": cameras,
            }
        )
    )

    rows = []
    for frame_index in range(2):
        row = {
            "frame_index": frame_index,
            "cam0_relpath": f"cam0/{frame_index:06d}.jpg",
            "cam1_relpath": f"cam1/{frame_index:06d}.jpg",
        }
        if depth:
            row["cam0_depth_relpath"] = f"cam0_depth/{frame_index:06d}.png"
        state = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        state.extend([0.01 * (joint + frame_index) for joint in range(6)])
        state.append(0.25 + frame_index * 0.5)
        action = [0.02 * (joint + frame_index) for joint in range(6)]
        action.append(0.2 + frame_index * 0.6)
        row.update(
            {f"state.{name}": value for name, value in zip(DEFAULT_STATE_NAMES, state)}
        )
        row.update(
            {f"action.{name}": value for name, value in zip(JOINT_ACTION_NAMES, action)}
        )
        rows.append(row)
        for directory in ("cam0", "cam1"):
            Image.fromarray(
                np.full((height, width, 3), frame_index, dtype=np.uint8)
            ).save(episode / directory / f"{frame_index:06d}.jpg")
        if depth:
            Image.fromarray(
                np.full((height, width), 1000 + frame_index, dtype=np.uint16)
            ).save(episode / "cam0_depth" / f"{frame_index:06d}.png")
    pd.DataFrame(rows).to_csv(episode / "frames.csv", index=False)
    return source


def test_discover_current_raw_dataset(tmp_path):
    source = make_raw_episode(tmp_path, depth=True)

    summary = discover_raw_dataset(source_root=source)

    assert summary.task == "pick cube"
    assert summary.total_frames == 2
    assert summary.episodes[0].state_names == DEFAULT_STATE_NAMES
    assert summary.episodes[0].action_names == JOINT_ACTION_NAMES
    assert [camera.name for camera in summary.episodes[0].camera_specs] == [
        "front",
        "wrist",
        "front_depth",
    ]


def test_discover_rejects_incomplete_or_old_episode(tmp_path):
    source = make_raw_episode(tmp_path)
    (source / "episode_001_incomplete").mkdir()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        discover_raw_dataset(source_root=source)

    (source / "episode_001_incomplete").rmdir()
    metadata = next(source.glob("episode_*/metadata.json"))
    payload = json.loads(metadata.read_text())
    payload["schema_version"] = "galaxea_a1_teleop_raw_v1"
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only.*raw_v3"):
        discover_raw_dataset(source_root=source)


def test_discover_rejects_cross_episode_camera_change(tmp_path):
    source = make_raw_episode(tmp_path, episode_index=0, width=5)
    make_raw_episode(tmp_path, episode_index=1, width=6)

    with pytest.raises(ValueError, match="camera contract changed"):
        discover_raw_dataset(source_root=source)


def test_iter_episode_frames_preserves_continuous_contract(tmp_path):
    source = make_raw_episode(tmp_path, depth=True)
    summary = discover_raw_dataset(source_root=source)
    episode = summary.episodes[0]
    contract = raw_episode_contract(
        state_names=episode.state_names,
        action_names=episode.action_names,
        camera_specs=episode.camera_specs,
    )

    frames = list(
        iter_episode_frames(episode=episode, task=summary.task, contract=contract)
    )

    assert frames[0]["observation.state"].shape == (14,)
    assert frames[0]["action"].shape == (7,)
    assert frames[0]["action"][-1] == pytest.approx(0.2)
    assert frames[1]["action"][-1] == pytest.approx(0.8)
    assert frames[0]["observation.images.front"].shape == (4, 5, 3)
    assert frames[0]["observation.images.front_depth"].shape == (4, 5, 1)
    assert frames[0]["observation.images.front_depth"].dtype == np.uint16


def test_failed_overwrite_preserves_previous_converted_dataset(tmp_path, monkeypatch):
    source = make_raw_episode(tmp_path)
    target = tmp_path / "converted"
    target.mkdir()
    (target / "complete.txt").write_text("previous")

    class FailingDataset:
        def add_frame(self, frame):
            del frame
            raise RuntimeError("conversion failed")

        def stop_image_writer(self):
            return None

    monkeypatch.setattr(
        convert_raw_module,
        "create_lerobot_dataset",
        lambda **_kwargs: FailingDataset(),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        convert_raw_dataset(
            source_root=source,
            target_root=target,
            repo_id="galaxea/test",
            overwrite=True,
        )

    assert (target / "complete.txt").read_text() == "previous"
    assert not any(".converted.staging-" in path.name for path in tmp_path.iterdir())


def test_current_raw_converts_with_real_lerobot_writer(tmp_path):
    source = make_raw_episode(tmp_path, width=64, height=48)
    target = tmp_path / "lerobot_v3"

    summary = convert_raw_dataset(
        source_root=source,
        target_root=target,
        repo_id="galaxea-a1/test_current_raw",
    )

    assert summary.total_frames == 2
    assert (target / "meta/info.json").is_file()
    assert (target / "meta/stats.json").is_file()
    assert len(list(target.glob("data/**/*.parquet"))) == 1
    assert len(list(target.glob("videos/**/*.mp4"))) == 2
