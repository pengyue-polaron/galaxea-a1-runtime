"""Asynchronous recording from an already-owned latest-frame camera reader."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VideoRecordingResult:
    path: Path
    frames: int
    fps: float
    elapsed_s: float
    warning: str | None = None


def recording_run_id(task_id: str, *, now: datetime | None = None) -> str:
    if not task_id or any(
        not (character.isalnum() or character in {"-", "_", "."})
        for character in task_id
    ):
        raise ValueError(
            f"recording task id contains unsupported characters: {task_id!r}"
        )
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{task_id}"


class LatestFrameVideoRecorder:
    """Encode a constant-rate H.264 MP4 without opening another camera handle."""

    def __init__(
        self,
        *,
        reader: Any,
        extract_bgr: Callable[[Any], np.ndarray],
        output_root: Path,
        run_id: str,
        width: int,
        height: int,
        fps: float,
        source: str,
        max_source_age_s: float,
        video_filename: str = "agent_view.mp4",
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        if min(width, height) <= 0 or width % 2 or height % 2:
            raise ValueError("video width and height must be positive even integers")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video fps must be finite and positive")
        if not np.isfinite(max_source_age_s) or max_source_age_s <= 0:
            raise ValueError("video max_source_age_s must be finite and positive")
        if not run_id or run_id.startswith(".") or "/" in run_id:
            raise ValueError(f"invalid recording run id: {run_id!r}")
        if (
            not video_filename
            or video_filename.startswith(".")
            or Path(video_filename).name != video_filename
            or "\\" in video_filename
            or not video_filename.endswith(".mp4")
            or len(video_filename.encode("utf-8")) > 240
        ):
            raise ValueError(f"invalid recording video filename: {video_filename!r}")
        self.reader = reader
        self.extract_bgr = extract_bgr
        self.output_root = output_root.expanduser().resolve()
        self.run_id = run_id
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.source = source
        self.video_filename = video_filename
        self.max_source_age_s = float(max_source_age_s)
        self.monotonic = monotonic
        self.final_dir = self.output_root / run_id
        self.staging_dir = self.output_root / f".{run_id}.staging"
        self.final_path = self.final_dir / self.video_filename
        self._staging_path = self.staging_dir / self.video_filename
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._frames = 0
        self._started_monotonic: float | None = None
        self._ended_monotonic: float | None = None
        self._started_wall_time = ""
        self._result: VideoRecordingResult | None = None

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def elapsed_s(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        end = self._ended_monotonic or self.monotonic()
        return max(0.0, end - self._started_monotonic)

    @property
    def result(self) -> VideoRecordingResult | None:
        return self._result

    def exception(self) -> BaseException | None:
        return self._error

    def start(self, *, startup_timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("AgentView recorder is already started")
        if self.final_dir.exists() or self.staging_dir.exists():
            raise FileExistsError(
                f"recording output already exists: {self.final_dir} or {self.staging_dir}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir()
        self._started_monotonic = self.monotonic()
        self._started_wall_time = datetime.now().astimezone().isoformat()
        self._thread = threading.Thread(
            target=self._record_loop,
            name="agent-view-recorder",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=startup_timeout_s):
            self._stop.set()
            self._thread.join(timeout=2.0)
            raise RuntimeError("AgentView video encoder did not start in time")
        if self._error is not None:
            raise RuntimeError(
                "AgentView video encoder failed to start"
            ) from self._error

    def close(self, *, timeout_s: float = 10.0) -> VideoRecordingResult | None:
        if self._result is not None:
            return self._result
        thread = self._thread
        if thread is None:
            return None
        self._stop.set()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise RuntimeError("AgentView video recorder did not stop")
        self._ended_monotonic = self.monotonic()
        warning = (
            None
            if self._error is None
            else f"{type(self._error).__name__}: {self._error}"
        )
        if self._frames <= 0 or not self._staging_path.is_file():
            raise RuntimeError(
                "AgentView recording produced no video frames"
            ) from self._error
        metadata = {
            "schema_version": 1,
            "video": self._staging_path.name,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self._frames,
            "elapsed_s": self.elapsed_s,
            "started_at": self._started_wall_time,
            "ended_at": datetime.now().astimezone().isoformat(),
            "warning": warning,
        }
        (self.staging_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(self.staging_dir, self.final_dir)
        self._result = VideoRecordingResult(
            path=self.final_path,
            frames=self._frames,
            fps=self.fps,
            elapsed_s=self.elapsed_s,
            warning=warning,
        )
        return self._result

    def _record_loop(self) -> None:
        container = None
        stream = None
        try:
            import av

            rate = Fraction(self.fps).limit_denominator(1000)
            container = av.open(str(self._staging_path), mode="w", format="mp4")
            stream = container.add_stream("libx264", rate=rate)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "20"}
            self._ready.set()
            interval = 1.0 / self.fps
            next_frame_at = self.monotonic()
            last_source_seq = -1
            image: np.ndarray | None = None
            while not self._stop.is_set():
                now = self.monotonic()
                remaining = next_frame_at - now
                if remaining > 0:
                    self._stop.wait(remaining)
                    continue
                next_frame_at += interval
                if next_frame_at < now - interval:
                    next_frame_at = now + interval

                reader_error = self._reader_exception()
                if reader_error is not None:
                    raise RuntimeError(
                        "AgentView camera reader failed"
                    ) from reader_error
                sample = self.reader.latest()
                if (
                    sample is None
                    or now - float(sample.monotonic_s) > self.max_source_age_s
                ):
                    continue
                if sample.seq != last_source_seq:
                    image = self._validated_image(self.extract_bgr(sample.value))
                    last_source_seq = sample.seq
                if image is None:
                    continue
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                self._frames += 1
        except BaseException as exc:  # The owning bridge surfaces this on close/read.
            self._error = exc
        finally:
            self._ready.set()
            if container is not None:
                try:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                    container.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc

    def _validated_image(self, value: np.ndarray) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.shape != (
            self.height,
            self.width,
            3,
        ):
            raise ValueError(
                "AgentView recording frame must have shape "
                f"({self.height}, {self.width}, 3), got "
                f"{getattr(value, 'shape', None)}"
            )
        if value.dtype != np.uint8:
            raise ValueError(
                f"AgentView recording frame must be uint8, got {value.dtype}"
            )
        return np.ascontiguousarray(value).copy()

    def _reader_exception(self) -> BaseException | None:
        getter = getattr(self.reader, "exception", None)
        return getter() if callable(getter) else None
