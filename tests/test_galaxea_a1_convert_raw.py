import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import galaxea_a1_runtime.lerobot.convert_raw as convert_raw_module
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.lerobot.convert_raw import (
    convert_raw_dataset,
    convert_raw_datasets,
    discover_raw_dataset,
    iter_episode_frames,
    raw_episode_contract,
)
from galaxea_a1_runtime.lerobot.eef_pack import pack_eef_v3_dataset
from galaxea_a1_runtime.lerobot.joint_pack import pack_joint_v3_dataset
from galaxea_a1_runtime.lerobot.pipeline import build_datasets
from galaxea_a1_runtime.lerobot.pipeline_config import load_pipeline_config
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset
from galaxea_a1_runtime.schema import (
    LEGACY_RAW_ACTION_NAMES,
    LEGACY_RAW_STATE_NAMES,
    JOINT_ACTION_NAMES_RAD,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    REPO_ROOT
    / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
)
PIPELINE_CONFIG_FIXTURE = REPO_ROOT / "tests/fixtures/dataset_pipeline.toml"
NO_TRIM = replace(
    load_pipeline_config(PIPELINE_CONFIG_FIXTURE).boundary_trim,
    enabled=False,
)


def make_raw_episode(
    root: Path,
    *,
    episode_index: int = 0,
    width: int = 5,
    height: int = 4,
    depth: bool = False,
    frame_count: int = 2,
    stationary_boundaries: bool = False,
    source_name: str = "raw_task",
    task: str = "pick cube",
) -> Path:
    source = root / source_name
    episode = source / f"episode_{episode_index:03d}_20260708_120000"
    for directory in ("cam0", "cam1", *(("cam0_depth",) if depth else ())):
        (episode / directory).mkdir(parents=True)
    (source / "task.txt").write_text(f"{task}\n")

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
                "task": task,
                "action_mode": "joint_absolute",
                "frame_count": frame_count,
                "fps_target": 30.0,
                "state_names": list(LEGACY_RAW_STATE_NAMES),
                "action_names": list(LEGACY_RAW_ACTION_NAMES),
                "cameras": cameras,
            }
        )
    )

    rows = []
    for frame_index in range(frame_count):
        row = {
            "frame_index": frame_index,
            "cam0_relpath": f"cam0/{frame_index:06d}.jpg",
            "cam1_relpath": f"cam1/{frame_index:06d}.jpg",
        }
        if depth:
            row["cam0_depth_relpath"] = f"cam0_depth/{frame_index:06d}.png"
        state = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        if stationary_boundaries:
            action_level = 0.0 if frame_index < 30 else 0.2
            if frame_index >= 250:
                action_level = 0.4
            state_level = 0.0 if frame_index < 35 else 0.2
            if frame_index >= 255:
                state_level = 0.4
            state.extend([state_level, 0.0, 0.0, 0.0, 0.0, 0.0])
            state.append(0.2)
            action = [action_level, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
        else:
            state.extend([0.01 * (joint + frame_index) for joint in range(6)])
            state.append(0.25 + frame_index * 0.5)
            action = [0.02 * (joint + frame_index) for joint in range(6)]
            action.append(0.2 + frame_index * 0.6)
        row.update(
            {
                f"state.{name}": value
                for name, value in zip(LEGACY_RAW_STATE_NAMES, state)
            }
        )
        row.update(
            {
                f"action.{name}": value
                for name, value in zip(LEGACY_RAW_ACTION_NAMES, action)
            }
        )
        rows.append(row)
        for directory in ("cam0", "cam1"):
            Image.fromarray(
                np.full((height, width, 3), frame_index % 256, dtype=np.uint8)
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
    assert summary.episodes[0].state_names == LEGACY_RAW_STATE_NAMES
    assert summary.episodes[0].action_names == LEGACY_RAW_ACTION_NAMES
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
            source_dataset="galaxea-a1/raw-v3",
            overwrite=True,
            trim_config=NO_TRIM,
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
        source_dataset="galaxea-a1/raw-v3",
        trim_config=NO_TRIM,
    )

    assert summary.total_frames == 2
    assert (target / "meta/info.json").is_file()
    assert (target / "meta/stats.json").is_file()
    trim = json.loads((target / "meta/trim.json").read_text())
    assert trim["policy"]["enabled"] is False
    assert trim["summary"]["trimmed_frames"] == 0
    assert len(list(target.glob("data/**/*.parquet"))) == 1
    assert len(list(target.glob("videos/**/*.mp4"))) == 2


def test_raw_conversion_rejects_absolute_source_dataset_id(tmp_path):
    source = make_raw_episode(tmp_path)

    with pytest.raises(ValueError, match="must not be an absolute path"):
        convert_raw_dataset(
            source_root=source,
            target_root=tmp_path / "lerobot_v3",
            repo_id="galaxea-a1/test_current_raw",
            source_dataset=str(source.resolve()),
            trim_config=NO_TRIM,
        )


def test_raw_conversion_applies_one_audited_boundary_trim(tmp_path, monkeypatch):
    source = make_raw_episode(
        tmp_path,
        frame_count=300,
        stationary_boundaries=True,
    )
    target = tmp_path / "trimmed"

    class RecordingDataset:
        def __init__(self, root):
            self.frames = []
            self.root = root
            (root / "meta").mkdir(parents=True)

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self, *, parallel_encoding):
            assert parallel_encoding is False
            return None

        def finalize(self):
            return None

        def stop_image_writer(self):
            return None

    datasets = []

    def create_dataset(*, config, contract):
        del contract
        dataset = RecordingDataset(config.root)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(convert_raw_module, "create_lerobot_dataset", create_dataset)
    trim_config = load_pipeline_config(PIPELINE_CONFIG_FIXTURE).boundary_trim

    convert_raw_dataset(
        source_root=source,
        target_root=target,
        repo_id="galaxea-a1/trimmed",
        source_dataset="galaxea-a1/raw-v3",
        trim_config=trim_config,
    )

    assert len(datasets[0].frames) == 263
    manifest = json.loads((target / "meta/trim.json").read_text())
    decision = manifest["episodes"][0]
    assert (decision["start"], decision["end"]) == (15, 278)
    assert manifest["summary"]["trimmed_frames"] == 37


def test_raw_conversion_combines_task_roots_without_losing_task_text(
    tmp_path, monkeypatch
):
    first_source = make_raw_episode(
        tmp_path, source_name="first_task", task="pick banana"
    )
    second_source = make_raw_episode(
        tmp_path, source_name="second_task", task="pick mango"
    )
    target = tmp_path / "combined"

    class RecordingDataset:
        def __init__(self, root):
            self.frames = []
            self.saved_episodes = 0
            (root / "meta").mkdir(parents=True)

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self, *, parallel_encoding):
            assert parallel_encoding is False
            self.saved_episodes += 1

        def finalize(self):
            return None

        def stop_image_writer(self):
            return None

    datasets = []

    def create_dataset(*, config, contract):
        del contract
        dataset = RecordingDataset(config.root)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(convert_raw_module, "create_lerobot_dataset", create_dataset)

    summaries = convert_raw_datasets(
        source_roots=(first_source, second_source),
        target_root=target,
        repo_id="galaxea-a1/combined",
        source_dataset="galaxea-a1/fruit-raw-v3",
        trim_config=NO_TRIM,
    )

    assert [summary.task for summary in summaries] == ["pick banana", "pick mango"]
    assert datasets[0].saved_episodes == 2
    assert [frame["task"] for frame in datasets[0].frames] == [
        "pick banana",
        "pick banana",
        "pick mango",
        "pick mango",
    ]
    manifest = json.loads((target / "meta/trim.json").read_text())
    assert manifest["source_dataset"] == "galaxea-a1/fruit-raw-v3"
    assert "source_datasets" not in manifest
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["tasks"] == ["pick banana", "pick mango"]
    assert [episode["episode"] for episode in manifest["episodes"]] == [
        "first_task/episode_000_20260708_120000",
        "second_task/episode_000_20260708_120000",
    ]


def test_v21_export_round_trips_through_official_lerobot_migrator(tmp_path):
    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts.convert_dataset_v21_to_v30 import (
        convert_dataset,
        legacy_load_episodes,
        legacy_load_episodes_stats,
        legacy_load_tasks,
        validate_local_dataset_version,
    )

    source = make_raw_episode(tmp_path, width=64, height=48)
    v3_root = tmp_path / "lerobot_v3"
    v21_root = tmp_path / "lerobot_v21"
    convert_raw_dataset(
        source_root=source,
        target_root=v3_root,
        repo_id="galaxea-a1/test_v3",
        source_dataset="galaxea-a1/raw-v3",
        trim_config=NO_TRIM,
    )

    result = export_v21_dataset(
        source_root=v3_root,
        target_root=v21_root,
        repo_id="galaxea-a1/test_v21",
    )

    validate_local_dataset_version(v21_root)
    assert result["format"] == "v2.1"
    assert result["episodes"] == 1
    assert result["frames"] == 2
    assert result["videos"] == 2
    assert legacy_load_episodes(v21_root)[0]["length"] == 2
    assert set(legacy_load_episodes_stats(v21_root)) == {0}
    assert legacy_load_tasks(v21_root)[0] == {0: "pick cube"}

    convert_dataset(
        repo_id="galaxea-a1/test_v21",
        root=v21_root,
        push_to_hub=False,
    )
    round_trip = LeRobotDataset(repo_id="galaxea-a1/test_v21", root=v21_root)
    assert len(round_trip) == 2
    assert round_trip.meta.total_episodes == 1
    assert round_trip.meta.info.codebase_version == "v3.0"


def test_joint_and_eef_outputs_are_model_agnostic_lerobot_datasets(tmp_path):
    raw = make_raw_episode(tmp_path, width=64, height=48)
    raw_v3 = tmp_path / "raw_v3"
    joint_v3 = tmp_path / "joint_v3"
    eef_v3 = tmp_path / "eef_v3"
    eef_v21 = tmp_path / "eef_v21"
    convert_raw_dataset(
        source_root=raw,
        target_root=raw_v3,
        repo_id="galaxea-a1/raw_v3",
        source_dataset="galaxea-a1/raw-v3",
        trim_config=NO_TRIM,
    )

    source_dataset = "galaxea-a1/raw-v3"
    joint_manifest = pack_joint_v3_dataset(
        source_root=raw_v3,
        target_root=joint_v3,
        repo_id="galaxea-a1/joint_v3",
        source_dataset=source_dataset,
    )
    eef_manifest = pack_eef_v3_dataset(
        source_root=raw_v3,
        target_root=eef_v3,
        urdf_path=URDF,
        repo_id="galaxea-a1/eef_v3",
        gripper_stroke_min_mm=0.0,
        gripper_stroke_max_mm=104.0,
        base_link="base_link",
        tip_link="arm_seg6",
        source_dataset=source_dataset,
    )
    eef_v21_manifest = export_v21_dataset(
        source_root=eef_v3,
        target_root=eef_v21,
        repo_id="galaxea-a1/eef_v21",
        source_dataset=source_dataset,
    )

    assert joint_manifest["format"] == "lerobot_v3_galaxea_a1_joint_absolute_v1"
    assert joint_manifest["representation"] == "joint"
    assert eef_manifest["format"] == ("lerobot_v3_galaxea_a1_eef_episode_relative_v1")
    assert eef_manifest["representation"] == "eef"
    assert eef_manifest["action"]["shape"] == [8]
    assert eef_manifest["source_dataset"] == source_dataset
    assert eef_manifest["source_format"] == TELEOP_RAW_SCHEMA_VERSION
    assert eef_manifest["kinematics"]["urdf"] == URDF.name
    assert not Path(eef_manifest["kinematics"]["urdf"]).is_absolute()
    assert str(URDF.parent) not in json.dumps(eef_manifest)
    eef_info = json.loads((eef_v3 / "meta/info.json").read_text())
    assert eef_info["features"]["observation.state"]["names"] == [
        *LEGACY_RAW_STATE_NAMES[:7],
        *JOINT_ACTION_NAMES_RAD,
    ]
    serialized_manifest = json.dumps(eef_manifest).lower()
    assert "lingbot" not in serialized_manifest
    assert "robotwin" not in serialized_manifest
    assert "recommended_policy" not in eef_manifest
    assert eef_v21_manifest["format"] == "v2.1"
    persisted = json.loads((eef_v21 / "meta/eef.json").read_text())
    assert persisted["format"] == ("lerobot_v2.1_galaxea_a1_eef_episode_relative_v1")
    assert persisted["source_dataset"] == source_dataset
    assert "source_v3_dataset" not in persisted
    assert persisted["conversion_intermediate"]["format"] == "lerobot_v3.0"
    assert (eef_v3 / "TRAINING.md").read_text().startswith("# A1 EEF LeRobot Dataset\n")
    assert (
        (eef_v21 / "TRAINING.md").read_text().startswith("# A1 EEF LeRobot Dataset\n")
    )


def test_pipeline_builds_four_independent_outputs_from_raw_v3(tmp_path):
    raw = make_raw_episode(tmp_path, width=64, height=48)
    summary = discover_raw_dataset(source_root=raw)
    first = summary.episodes[0]
    source_contract = raw_episode_contract(
        state_names=first.state_names,
        action_names=first.action_names,
        camera_specs=first.camera_specs,
    )
    source_dataset = "galaxea-a1/test-raw-v3"
    config = replace(
        load_pipeline_config(PIPELINE_CONFIG_FIXTURE),
        raw_source_id=source_dataset,
        raw_source_roots=(raw,),
        source_contract=source_contract,
        joint_v3_target_root=tmp_path / "joint_v3",
        joint_v3_archive_path=tmp_path / "joint_v3.tar.gz",
        joint_v21_target_root=tmp_path / "joint_v21",
        joint_v21_archive_path=tmp_path / "joint_v21.tar.gz",
        eef_v3_target_root=tmp_path / "eef_v3",
        eef_v3_archive_path=tmp_path / "eef_v3.tar.gz",
        eef_v21_target_root=tmp_path / "eef_v21",
        eef_v21_archive_path=tmp_path / "eef_v21.tar.gz",
    )

    result = build_datasets(config)

    assert set(result) == {"joint-v3", "joint-v2.1", "eef-v3", "eef-v2.1"}
    manifests = [
        json.loads((tmp_path / output / "meta" / filename).read_text())
        for output, filename in (
            ("joint_v3", "joint.json"),
            ("joint_v21", "joint.json"),
            ("eef_v3", "eef.json"),
            ("eef_v21", "eef.json"),
        )
    ]
    assert all(manifest["source_dataset"] == source_dataset for manifest in manifests)
    assert all(
        manifest["source_format"] == TELEOP_RAW_SCHEMA_VERSION for manifest in manifests
    )
    trim_manifests = [
        json.loads((tmp_path / output / "meta/trim.json").read_text())
        for output in ("joint_v3", "joint_v21", "eef_v3", "eef_v21")
    ]
    assert all(manifest == trim_manifests[0] for manifest in trim_manifests)
    assert trim_manifests[0]["policy"]["enabled"] is True
    assert trim_manifests[0]["summary"]["trimmed_frames"] == 0
    assert trim_manifests[0]["source_dataset"] == source_dataset
    assert str(raw.resolve()) not in json.dumps(trim_manifests[0])
    for v3_name, v21_name in (("joint_v3", "joint_v21"), ("eef_v3", "eef_v21")):
        v3_frames = pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted((tmp_path / v3_name).glob("data/**/*.parquet"))
            ],
            ignore_index=True,
        )
        v21_frames = pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted((tmp_path / v21_name).glob("data/**/*.parquet"))
            ],
            ignore_index=True,
        )
        np.testing.assert_allclose(
            np.stack(v21_frames["observation.state"]),
            np.stack(v3_frames["observation.state"]),
        )
        np.testing.assert_allclose(
            np.stack(v21_frames["action"]),
            np.stack(v3_frames["action"]),
        )
    joint_actions = np.stack(
        pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted((tmp_path / "joint_v3").glob("data/**/*.parquet"))
            ],
            ignore_index=True,
        )["action"]
    )
    np.testing.assert_allclose(joint_actions[:, -1], [0.2, 0.8])
    assert ".a1-raw-v3-conversion-" not in json.dumps(manifests)
    assert not list(tmp_path.glob(".a1-raw-v3-conversion-*"))
