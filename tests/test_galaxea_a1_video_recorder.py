from datetime import datetime, timezone
from pathlib import Path
import time

import av
import numpy as np

from galaxea_a1_runtime.hardware.cameras import CameraSample
from galaxea_a1_runtime.hardware.video_recorder import (
    LatestFrameVideoRecorder,
    recording_run_id,
)


class FakeReader:
    def __init__(self, image: np.ndarray):
        self.sample = CameraSample(seq=1, monotonic_s=time.perf_counter(), value=image)

    def latest(self) -> CameraSample:
        self.sample = CameraSample(
            seq=self.sample.seq + 1,
            monotonic_s=time.perf_counter(),
            value=self.sample.value,
        )
        return self.sample

    def exception(self):
        return None


def test_agent_view_video_is_atomically_finalized_as_playable_mp4(tmp_path: Path):
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    recorder = LatestFrameVideoRecorder(
        reader=FakeReader(image),
        extract_bgr=lambda value: value,
        output_root=tmp_path,
        run_id="run",
        width=64,
        height=48,
        fps=20.0,
        source="agent-test",
        max_source_age_s=0.5,
    )

    recorder.start()
    time.sleep(0.16)
    result = recorder.close()

    assert result is not None
    assert result.path == tmp_path / "run/agent_view.mp4"
    assert result.frames >= 2
    assert result.path.is_file()
    assert not (tmp_path / ".run.staging").exists()
    with av.open(str(result.path)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == result.frames
    assert recorder.close() == result


def test_recording_run_id_is_timestamped_and_task_scoped():
    now = datetime(2026, 7, 18, 1, 2, 3, 456789, tzinfo=timezone.utc)

    assert recording_run_id("banana_blue_plate", now=now) == (
        "20260718_010203_456789_banana_blue_plate"
    )
