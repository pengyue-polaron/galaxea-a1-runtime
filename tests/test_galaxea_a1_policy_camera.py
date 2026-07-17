from pathlib import Path

from galaxea_a1_runtime.apps import policy_camera
from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from galaxea_a1_runtime.hardware.video_recorder import VideoRecordingResult


def test_policy_camera_finalizes_recording_before_preview_and_cameras(
    tmp_path: Path, monkeypatch
):
    operations: list[str] = []
    result = VideoRecordingResult(
        path=tmp_path / "agent_view.mp4",
        frames=10,
        fps=30.0,
        elapsed_s=1.0,
    )

    class Recorder:
        def close(self):
            operations.append("recording")
            return result

    class Preview:
        def close(self):
            operations.append("preview")

    monkeypatch.setattr(
        policy_camera,
        "close_camera_resources",
        lambda *_args: operations.append("cameras"),
    )
    session = PolicyCameraSession.__new__(PolicyCameraSession)
    session.agent_recorder = Recorder()
    session.recording_result = None
    session.preview = Preview()
    session.wrist_reader = object()
    session.front_reader = object()
    session.wrist_camera = object()
    session.front_camera = object()

    session.close()

    assert operations == ["recording", "preview", "cameras"]
    assert session.recording_result == result
