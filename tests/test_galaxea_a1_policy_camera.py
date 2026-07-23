from pathlib import Path

from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from galaxea_a1_runtime.hardware.video_recorder import PairedVideoRecordingResult


def test_policy_camera_finalizes_recording_before_bridge_disconnect(
    tmp_path: Path,
):
    operations: list[str] = []
    result = PairedVideoRecordingResult(
        front_path=tmp_path / "front.mp4",
        wrist_path=tmp_path / "wrist.mp4",
        timeline_path=tmp_path / "camera_timeline.jsonl",
        metadata_path=tmp_path / "camera_recording.json",
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
    session.camera_recorder = Recorder()
    session.recording_result = None
    session.camera_bridge = Bridge()
    session.wrist_reader = object()
    session.front_reader = object()

    session.close()

    assert operations == ["recording", "bridge"]
    assert session.recording_result == result
