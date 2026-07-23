import json
from pathlib import Path
import time

import av
import numpy as np
import pytest

from galaxea_a1_runtime.hardware.camera_reader import CameraSample
from galaxea_a1_runtime.hardware.video_recorder import PairedCameraVideoRecorder


class FakePairReader:
    def __init__(self, front: np.ndarray, wrist: np.ndarray):
        now = time.perf_counter()
        self.front = CameraSample(seq=1, monotonic_s=now, value=front)
        self.wrist = CameraSample(seq=1, monotonic_s=now, value=wrist)

    def latest_pair(self) -> tuple[CameraSample, CameraSample]:
        now = time.perf_counter()
        self.front = CameraSample(
            seq=self.front.seq + 1,
            monotonic_s=now,
            value=self.front.value,
        )
        self.wrist = CameraSample(
            seq=self.wrist.seq + 1,
            monotonic_s=now,
            value=self.wrist.value,
        )
        return self.front, self.wrist

    def exception(self):
        return None


def _recorder(tmp_path: Path) -> PairedCameraVideoRecorder:
    front = np.full((48, 64, 3), 120, dtype=np.uint8)
    wrist = np.full((24, 32, 3), 80, dtype=np.uint8)
    reader = FakePairReader(front, wrist)
    return PairedCameraVideoRecorder(
        read_pair=reader.latest_pair,
        reader_exception=reader.exception,
        extract_front_bgr=lambda value: value,
        extract_wrist_bgr=lambda value: value,
        output_root=tmp_path,
        run_id="run",
        front_width=64,
        front_height=48,
        wrist_width=32,
        wrist_height=24,
        fps=20.0,
        front_source="front-test",
        wrist_source="wrist-test",
        max_source_age_s=0.5,
        max_pair_skew_s=0.02,
        front_video_filename="front.mp4",
        wrist_video_filename="wrist.mp4",
    )


def test_paired_camera_videos_and_timeline_are_atomically_finalized(
    tmp_path: Path,
):
    recorder = _recorder(tmp_path)

    recorder.start()
    time.sleep(0.16)
    result = recorder.close()

    assert result is not None
    assert result.front_path == tmp_path / "run/front.mp4"
    assert result.wrist_path == tmp_path / "run/wrist.mp4"
    assert result.frames >= 2
    assert not (tmp_path / ".run.staging").exists()
    for path in (result.front_path, result.wrist_path):
        with av.open(str(path)) as container:
            decoded = list(container.decode(video=0))
        assert len(decoded) == result.frames
    timeline = [
        json.loads(line) for line in result.timeline_path.read_text().splitlines()
    ]
    assert len(timeline) == result.frames
    assert timeline[0]["front_seq"] == timeline[0]["wrist_seq"]
    recording = json.loads(result.metadata_path.read_text())
    assert recording["videos"]["front"]["file"] == "front.mp4"
    assert recording["videos"]["wrist"]["file"] == "wrist.mp4"
    assert recording["timeline"] == "camera_timeline.jsonl"
    assert recording["frames"] == result.frames
    assert recorder.close() == result


def test_paired_camera_recording_rejects_filename_over_filesystem_byte_budget(
    tmp_path: Path,
):
    front = np.full((48, 64, 3), 120, dtype=np.uint8)
    wrist = np.full((24, 32, 3), 80, dtype=np.uint8)
    reader = FakePairReader(front, wrist)

    with pytest.raises(ValueError, match="recording output filename"):
        PairedCameraVideoRecorder(
            read_pair=reader.latest_pair,
            reader_exception=reader.exception,
            extract_front_bgr=lambda value: value,
            extract_wrist_bgr=lambda value: value,
            output_root=tmp_path,
            run_id="named-run",
            front_width=64,
            front_height=48,
            wrist_width=32,
            wrist_height=24,
            fps=20.0,
            front_source="front-test",
            wrist_source="wrist-test",
            max_source_age_s=0.5,
            max_pair_skew_s=0.02,
            front_video_filename=f"{'场' * 100}.mp4",
            wrist_video_filename="wrist.mp4",
        )


def test_paired_camera_recording_never_publishes_one_valid_and_one_invalid_stream(
    tmp_path: Path,
):
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.extract_wrist_bgr = lambda _value: np.zeros((10, 10, 3), dtype=np.uint8)

    time.sleep(0.08)
    with pytest.raises(RuntimeError, match="paired camera video recording failed"):
        recorder.close()

    assert not (tmp_path / "run").exists()
    assert (tmp_path / ".run.staging").is_dir()
