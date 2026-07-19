from pathlib import Path

from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from galaxea_a1_runtime.hardware.video_recorder import VideoRecordingResult


def test_policy_camera_finalizes_recording_before_bridge_disconnect(
    tmp_path: Path,
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

    class Bridge:
        def close(self):
            operations.append("bridge")

    session = PolicyCameraSession.__new__(PolicyCameraSession)
    session.agent_recorder = Recorder()
    session.recording_result = None
    session.camera_bridge = Bridge()
    session.wrist_reader = object()
    session.front_reader = object()

    session.close()

    assert operations == ["recording", "bridge"]
    assert session.recording_result == result
