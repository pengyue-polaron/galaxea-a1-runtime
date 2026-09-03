import sys
from types import SimpleNamespace

from embodied_ops import LeadingStillnessConfig

from galaxea_a1_runtime.apps.teleop.collection_task import normalize_collection_task
from galaxea_a1_runtime.apps.teleop.recording import CapturedFrame, record_episode
from galaxea_a1_runtime.collection import (
    find_joint_action_step_violation,
)


def test_collection_tasks_are_normalized_per_episode():
    assert normalize_collection_task("  pick fruit  ") == "pick fruit"
    assert normalize_collection_task("place fruit") == "place fruit"


def test_joint_action_quality_check_rejects_discontinuity():
    violation = find_joint_action_step_violation(
        [(0.0, 0.0), (0.1, 0.2), (0.15, 1.0)],
        action_names=("joint_1", "joint_2"),
        max_step_rad=0.35,
    )

    assert violation is not None
    assert violation.frame_index == 2
    assert violation.joint_name == "joint_2"
    assert violation.step_rad == 0.8


def test_joint_action_quality_check_accepts_continuous_actions():
    assert (
        find_joint_action_step_violation(
            [(0.0, 0.0, 0.0), (0.1, -0.1, 1.0), (0.2, -0.2, 0.0)],
            action_names=("joint_1", "joint_2", "gripper"),
            max_step_rad=0.35,
        )
        is None
    )


def test_collection_recording_trims_stationary_prefix_and_keeps_preroll(monkeypatch):
    readiness_events = []
    actions = iter(
        (
            (0.0,) * 7,
            (0.0,) * 7,
            (0.01,) * 7,
            (0.03,) * 7,
            (0.04,) * 7,
            (0.05,) * 7,
        )
    )
    commands = iter((None, None, None, None, None, ""))

    def capture(_recorder, _last_camera_seq):
        assert readiness_events == ["fresh-cameras", "recording-ready"]
        action = next(actions)
        return CapturedFrame(
            values={"action": action},
            action=action,
            camera_seq={"front": 1, "wrist": 1},
        )

    class Reader:
        def __init__(self, name):
            self.name = name

        def latest_seq(self):
            return 0

    class Dataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, values):
            self.frames.append(values)

    monkeypatch.setitem(
        sys.modules, "rospy", SimpleNamespace(is_shutdown=lambda: False)
    )
    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.teleop.recording.wait_for_new_camera_samples",
        lambda *_args, **_kwargs: readiness_events.append("fresh-cameras"),
    )
    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.teleop.recording._FrameRecorder.capture",
        capture,
    )
    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.teleop.recording._poll_stdin_line",
        lambda: next(commands),
    )
    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.teleop.recording.time.perf_counter",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.teleop.recording.time.sleep",
        lambda _seconds: None,
    )
    dataset = Dataset()

    recorded = record_episode(
        episode_index=0,
        dataset=dataset,
        task="place fruit",
        front_reader=Reader("front"),
        wrist_reader=Reader("wrist"),
        ros_state=object(),
        fps=30.0,
        max_duration_s=0.0,
        depth_enabled=False,
        front_crop=None,
        camera_ready_timeout_s=1.0,
        max_camera_age_s=0.5,
        max_camera_pair_skew_s=0.1,
        leading_stillness=LeadingStillnessConfig(
            enabled=True,
            action_thresholds=(0.02,) * 7,
            reference_frames=2,
            motion_frames=2,
            preroll_frames=1,
        ),
        on_ready=lambda: readiness_events.append("recording-ready"),
    )

    assert readiness_events == ["fresh-cameras", "recording-ready"]
    assert recorded.sampled_frame_count == 5
    assert recorded.frame_count == 3
    assert recorded.trimmed_frame_count == 2
    assert [frame["action"] for frame in dataset.frames] == [
        (0.01,) * 7,
        (0.03,) * 7,
        (0.04,) * 7,
    ]
