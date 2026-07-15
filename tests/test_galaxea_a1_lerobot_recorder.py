from pathlib import Path

import pytest

from galaxea_a1_runtime.lerobot.dataset import DatasetConfig
from galaxea_a1_runtime.hardware.io import A1Observation, InMemoryA1HardwareIO
from galaxea_a1_runtime.lerobot.recorder import LeRobotEpisodeRecorder
from galaxea_a1_runtime.lerobot.writer import LeRobotV3DatasetWriter
from galaxea_a1_runtime.policies.actions import normalize_action
from galaxea_a1_runtime.schema import ActionMode, CameraSpec, default_dataset_contract


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0
        self.finalized = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        self.finalized += 1


def build_recorder():
    contract = default_dataset_contract(
        action_mode=ActionMode.EEF_TRANSLATION,
        cameras=(CameraSpec("front", height=480, width=640),),
    )
    dataset = FakeDataset()
    writer = LeRobotV3DatasetWriter(
        config=DatasetConfig(repo_id="galaxea/a1_test", root=Path("/tmp/a1"), fps=20),
        contract=contract,
        dataset=dataset,
    )
    io = InMemoryA1HardwareIO(
        A1Observation(
            state=(0.0,) * 14,
            images={"front": "image"},
            timestamp=2.5,
        )
    )
    return LeRobotEpisodeRecorder(io=io, writer=writer, task="pick cube"), io, dataset


def test_episode_recorder_records_passively_by_default():
    recorder, io, dataset = build_recorder()
    action = normalize_action([0.01, 0.0, 0.0, 0.5], mode=ActionMode.EEF_TRANSLATION)

    recorder.open()
    step = recorder.record_step(action)
    recorder.save_episode()
    recorder.close(disconnect_io=True)

    assert step.executed is False
    assert io.last_action is None
    assert dataset.frames[0]["observation.images.front"] == "image"
    assert dataset.frames[0]["action"] == pytest.approx((0.01, 0.0, 0.0, 0.5))
    assert dataset.saved == 1
    assert dataset.finalized == 1
    assert io.is_connected is False


def test_episode_recorder_executes_only_when_requested():
    recorder, io, dataset = build_recorder()
    action = normalize_action([0.01, 0.0, 0.0, 0.5], mode=ActionMode.EEF_TRANSLATION)

    recorder.open()
    step = recorder.record_step(action, execute=True)

    assert step.executed is True
    assert io.last_action is action
    assert dataset.frames[0]["action"] == pytest.approx(action.values)


def test_episode_recorder_validates_before_execution():
    contract = default_dataset_contract(
        action_mode=ActionMode.EEF_TRANSLATION,
        cameras=(CameraSpec("front", height=480, width=640),),
    )
    writer = LeRobotV3DatasetWriter(
        config=DatasetConfig(repo_id="galaxea/a1_test", root=Path("/tmp/a1"), fps=20),
        contract=contract,
        dataset=FakeDataset(),
    )
    io = InMemoryA1HardwareIO(A1Observation(state=(0.0,) * 14, images={}))
    recorder = LeRobotEpisodeRecorder(io=io, writer=writer, task="pick cube")
    action = normalize_action([0.01, 0.0, 0.0, 0.5], mode=ActionMode.EEF_TRANSLATION)

    recorder.open()
    with pytest.raises(ValueError, match="front"):
        recorder.record_step(action, execute=True)

    assert io.last_action is None
