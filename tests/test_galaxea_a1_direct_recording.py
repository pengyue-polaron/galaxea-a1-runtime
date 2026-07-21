import json
from pathlib import Path

import numpy as np
import pytest

import embodied_ops.artifacts as artifacts
from embodied_ops import PublishedOutputCleanupError

import galaxea_a1_runtime.lerobot.dataset_package as dataset_package
from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectDatasetIdentity,
    DirectLeRobotEpisode,
    dataset_repo_id,
    inspect_direct_dataset,
)
from galaxea_a1_runtime.schema import CameraSpec, canonical_dataset_contract


def _frame(value: float) -> dict:
    return {
        "observation.state": np.full(14, value, dtype=np.float32),
        "action": np.full(7, value, dtype=np.float32),
        "observation.images.front": np.full((64, 64, 3), 10, dtype=np.uint8),
        "observation.images.wrist": np.full((64, 64, 3), 20, dtype=np.uint8),
        "task": "pick cube",
    }


def _episode(root: Path) -> DirectLeRobotEpisode:
    return DirectLeRobotEpisode(
        identity=DirectDatasetIdentity(
            target_root=root,
            repo_id="galaxea-a1/direct-test",
            fps=30,
            contract=canonical_dataset_contract(
                cameras=(
                    CameraSpec("front", height=64, width=64),
                    CameraSpec("wrist", height=64, width=64),
                )
            ),
            experiment="direct-test",
        ),
        task="pick cube",
        provenance={"quality_checks": {"max_camera_pair_skew_s": 0.1}},
    )


def test_direct_lerobot_dataset_records_and_atomically_appends(tmp_path: Path):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.add_frame(_frame(0.2))
        episode.commit()

    first_data = next((root / "data").rglob("*.parquet"))
    first_inode = first_data.stat().st_ino
    first_payload = first_data.read_bytes()

    with _episode(root) as episode:
        episode.add_frame(_frame(0.3))
        episode.commit()

    state = inspect_direct_dataset(
        _episode(root).identity,
        expected_task="pick cube",
    )
    assert (state.total_episodes, state.total_frames) == (2, 3)
    assert first_data.stat().st_ino == first_inode
    assert first_data.read_bytes() == first_payload

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id="galaxea-a1/direct-test", root=root)
    assert len(dataset) == 3
    assert dataset.meta.features["action"]["names"][-1] == "gripper_normalized"
    assert dataset.meta.features["observation.images.front"]["dtype"] == "video"
    assert len(list((root / "videos").rglob("*.mp4"))) == 4


def test_discard_preserves_the_previous_complete_dataset(tmp_path: Path):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()
    before = json.loads((root / "meta/info.json").read_text())

    with _episode(root) as episode:
        episode.add_frame(_frame(0.2))

    assert json.loads((root / "meta/info.json").read_text()) == before
    assert not list(tmp_path.glob(".direct-test.staging-*"))


def test_failed_episode_save_preserves_the_previous_complete_dataset(tmp_path: Path):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()
    before = (root / "meta/info.json").read_bytes()

    with pytest.raises(RuntimeError, match="synthetic save failure"):
        with _episode(root) as episode:
            episode.add_frame(_frame(0.2))

            def fail_save(*, parallel_encoding):
                assert parallel_encoding is False
                raise RuntimeError("synthetic save failure")

            episode._dataset.save_episode = fail_save
            episode.commit()

    assert (root / "meta/info.json").read_bytes() == before
    assert not list(tmp_path.glob(".direct-test.staging-*"))


def test_published_episode_remains_committed_when_backup_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()

    original_remove = artifacts._remove

    def fail_backup_cleanup(path):
        if ".backup-" in path.name:
            raise PermissionError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(artifacts, "_remove", fail_backup_cleanup)
    with pytest.raises(PublishedOutputCleanupError, match="published output") as error:
        with _episode(root) as episode:
            episode.add_frame(_frame(0.2))
            episode.commit()

    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (2, 2)
    assert error.value.backup.is_dir()


def test_append_requires_snapshot_hardlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()
    before = (root / "meta/info.json").read_bytes()

    def fail_link(*_args, **_kwargs):
        raise OSError("unsupported")

    monkeypatch.setattr(dataset_package.os, "link", fail_link)

    with pytest.raises(RuntimeError, match="requires hard-link support"):
        with _episode(root):
            raise AssertionError("must fail before recording")

    assert (root / "meta/info.json").read_bytes() == before
    assert not list(tmp_path.glob(".direct-test.staging-*"))


def test_append_rejects_collection_provenance_drift(tmp_path: Path):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()

    changed = _episode(root)
    changed.provenance["quality_checks"] = {"max_camera_pair_skew_s": 0.2}
    with pytest.raises(ValueError, match="provenance changed"):
        with changed:
            raise AssertionError("must reject before recording")


@pytest.mark.parametrize("kind", ["staging", "backup"])
def test_interrupted_transaction_blocks_collection_for_inspection(
    tmp_path: Path,
    kind: str,
):
    root = tmp_path / "direct-test"
    (tmp_path / f".direct-test.{kind}-crash").mkdir()

    with pytest.raises(ValueError, match="unfinished transaction"):
        inspect_direct_dataset(
            _episode(root).identity,
        )


def test_inspection_rejects_a_missing_referenced_payload(tmp_path: Path):
    root = tmp_path / "direct-test"
    with _episode(root) as episode:
        episode.add_frame(_frame(0.1))
        episode.commit()
    next((root / "data").rglob("*.parquet")).unlink()

    with pytest.raises(ValueError, match="data payload is missing"):
        inspect_direct_dataset(_episode(root).identity)


def test_dataset_repo_id_uses_the_tracked_prefix():
    assert (
        dataset_repo_id("pengyue-polaron/galaxea-a1", "pick-cube")
        == "pengyue-polaron/galaxea-a1-pick-cube"
    )
