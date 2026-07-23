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

from galaxea_a1_runtime.hardware.camera_reader import CameraSample


@dataclass(frozen=True)
class PairedVideoRecordingResult:
    front_path: Path
    wrist_path: Path
    timeline_path: Path
    metadata_path: Path
    frames: int
    fps: float
    elapsed_s: float


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


class PairedCameraVideoRecorder:
    """Encode synchronized front/wrist H.264 MP4s in one atomic transaction."""

    def __init__(
        self,
        *,
        read_pair: Callable[[], tuple[CameraSample, CameraSample] | None],
        reader_exception: Callable[[], BaseException | None],
        extract_front_bgr: Callable[[Any], np.ndarray],
        extract_wrist_bgr: Callable[[Any], np.ndarray],
        output_root: Path,
        run_id: str,
        front_width: int,
        front_height: int,
        wrist_width: int,
        wrist_height: int,
        fps: float,
        front_source: str,
        wrist_source: str,
        max_source_age_s: float,
        max_pair_skew_s: float,
        front_video_filename: str,
        wrist_video_filename: str,
        timeline_filename: str = "camera_timeline.jsonl",
        metadata_filename: str = "camera_recording.json",
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        for label, width, height in (
            ("front", front_width, front_height),
            ("wrist", wrist_width, wrist_height),
        ):
            if min(width, height) <= 0 or width % 2 or height % 2:
                raise ValueError(
                    f"{label} video width and height must be positive even integers"
                )
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video fps must be finite and positive")
        if not np.isfinite(max_source_age_s) or max_source_age_s <= 0:
            raise ValueError("video max_source_age_s must be finite and positive")
        if not np.isfinite(max_pair_skew_s) or max_pair_skew_s < 0:
            raise ValueError("video max_pair_skew_s must be finite and non-negative")
        if not run_id or run_id.startswith(".") or "/" in run_id:
            raise ValueError(f"invalid recording run id: {run_id!r}")
        _validate_output_filename(front_video_filename, suffix=".mp4")
        _validate_output_filename(wrist_video_filename, suffix=".mp4")
        _validate_output_filename(timeline_filename, suffix=".jsonl")
        _validate_output_filename(metadata_filename, suffix=".json")
        if (
            len(
                {
                    front_video_filename,
                    wrist_video_filename,
                    timeline_filename,
                    metadata_filename,
                }
            )
            != 4
        ):
            raise ValueError("camera recording output filenames must be distinct")
        self.read_pair = read_pair
        self.reader_exception = reader_exception
        self.extract_front_bgr = extract_front_bgr
        self.extract_wrist_bgr = extract_wrist_bgr
        self.output_root = output_root.expanduser().resolve()
        self.run_id = run_id
        self.front_width = int(front_width)
        self.front_height = int(front_height)
        self.wrist_width = int(wrist_width)
        self.wrist_height = int(wrist_height)
        self.fps = float(fps)
        self.front_source = front_source
        self.wrist_source = wrist_source
        self.max_source_age_s = float(max_source_age_s)
        self.max_pair_skew_s = float(max_pair_skew_s)
        self.monotonic = monotonic
        self.final_dir = self.output_root / run_id
        self.staging_dir = self.output_root / f".{run_id}.staging"
        self.front_path = self.final_dir / front_video_filename
        self.wrist_path = self.final_dir / wrist_video_filename
        self.timeline_path = self.final_dir / timeline_filename
        self.metadata_path = self.final_dir / metadata_filename
        self._front_staging_path = self.staging_dir / front_video_filename
        self._wrist_staging_path = self.staging_dir / wrist_video_filename
        self._timeline_staging_path = self.staging_dir / timeline_filename
        self._metadata_staging_path = self.staging_dir / metadata_filename
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._frames = 0
        self._started_monotonic: float | None = None
        self._ended_monotonic: float | None = None
        self._started_wall_time = ""
        self._result: PairedVideoRecordingResult | None = None

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
    def result(self) -> PairedVideoRecordingResult | None:
        return self._result

    def exception(self) -> BaseException | None:
        return self._error

    def start(self, *, startup_timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("paired camera recorder is already started")
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
            name="paired-camera-recorder",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=startup_timeout_s):
            self._stop.set()
            self._thread.join(timeout=2.0)
            raise RuntimeError("paired camera video encoders did not start in time")
        if self._error is not None:
            raise RuntimeError("paired camera video encoders failed to start") from (
                self._error
            )

    def close(self, *, timeout_s: float = 10.0) -> PairedVideoRecordingResult | None:
        if self._result is not None:
            return self._result
        thread = self._thread
        if thread is None:
            return None
        self._stop.set()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise RuntimeError("paired camera video recorder did not stop")
        self._ended_monotonic = self.monotonic()
        if self._error is not None:
            raise RuntimeError("paired camera video recording failed") from self._error
        required = (
            self._front_staging_path,
            self._wrist_staging_path,
            self._timeline_staging_path,
        )
        if self._frames <= 0 or any(not path.is_file() for path in required):
            raise RuntimeError(
                "paired camera recording produced no complete video frames"
            )
        metadata = {
            "schema_version": 1,
            "videos": {
                "front": {
                    "file": self._front_staging_path.name,
                    "source": self.front_source,
                    "width": self.front_width,
                    "height": self.front_height,
                },
                "wrist": {
                    "file": self._wrist_staging_path.name,
                    "source": self.wrist_source,
                    "width": self.wrist_width,
                    "height": self.wrist_height,
                },
            },
            "timeline": self._timeline_staging_path.name,
            "fps": self.fps,
            "frames": self._frames,
            "elapsed_s": self.elapsed_s,
            "started_at": self._started_wall_time,
            "ended_at": datetime.now().astimezone().isoformat(),
            "max_source_age_s": self.max_source_age_s,
            "max_pair_skew_s": self.max_pair_skew_s,
        }
        self._metadata_staging_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(self.staging_dir, self.final_dir)
        self._result = PairedVideoRecordingResult(
            front_path=self.front_path,
            wrist_path=self.wrist_path,
            timeline_path=self.timeline_path,
            metadata_path=self.metadata_path,
            frames=self._frames,
            fps=self.fps,
            elapsed_s=self.elapsed_s,
        )
        return self._result

    def _record_loop(self) -> None:
        front_container = None
        front_stream = None
        wrist_container = None
        wrist_stream = None
        timeline = None
        try:
            import av

            rate = Fraction(self.fps).limit_denominator(1000)
            front_container, front_stream = _open_h264_stream(
                av,
                self._front_staging_path,
                rate=rate,
                width=self.front_width,
                height=self.front_height,
            )
            wrist_container, wrist_stream = _open_h264_stream(
                av,
                self._wrist_staging_path,
                rate=rate,
                width=self.wrist_width,
                height=self.wrist_height,
            )
            timeline = self._timeline_staging_path.open("w", encoding="utf-8")
            self._ready.set()
            interval = 1.0 / self.fps
            next_frame_at = self.monotonic()
            last_pair = (-1, -1)
            front_image: np.ndarray | None = None
            wrist_image: np.ndarray | None = None
            front_sample: CameraSample | None = None
            wrist_sample: CameraSample | None = None
            while not self._stop.is_set():
                now = self.monotonic()
                remaining = next_frame_at - now
                if remaining > 0:
                    self._stop.wait(remaining)
                    continue
                next_frame_at += interval
                if next_frame_at < now - interval:
                    next_frame_at = now + interval

                reader_error = self.reader_exception()
                if reader_error is not None:
                    raise RuntimeError("camera bridge reader failed") from reader_error
                pair = self.read_pair()
                if pair is None:
                    continue
                current_front, current_wrist = pair
                if (
                    now - float(current_front.monotonic_s) > self.max_source_age_s
                    or now - float(current_wrist.monotonic_s) > self.max_source_age_s
                ):
                    continue
                skew_s = abs(current_front.monotonic_s - current_wrist.monotonic_s)
                if skew_s > self.max_pair_skew_s:
                    continue
                pair_id = (current_front.seq, current_wrist.seq)
                if pair_id != last_pair:
                    front_image = self._validated_image(
                        self.extract_front_bgr(current_front.value),
                        width=self.front_width,
                        height=self.front_height,
                        label="front",
                    )
                    wrist_image = self._validated_image(
                        self.extract_wrist_bgr(current_wrist.value),
                        width=self.wrist_width,
                        height=self.wrist_height,
                        label="wrist",
                    )
                    front_sample = current_front
                    wrist_sample = current_wrist
                    last_pair = pair_id
                if (
                    front_image is None
                    or wrist_image is None
                    or front_sample is None
                    or wrist_sample is None
                ):
                    continue
                front_frame = av.VideoFrame.from_ndarray(front_image, format="bgr24")
                wrist_frame = av.VideoFrame.from_ndarray(wrist_image, format="bgr24")
                for packet in front_stream.encode(front_frame):
                    front_container.mux(packet)
                for packet in wrist_stream.encode(wrist_frame):
                    wrist_container.mux(packet)
                timeline.write(
                    json.dumps(
                        {
                            "frame_index": self._frames,
                            "front_seq": front_sample.seq,
                            "front_monotonic_s": front_sample.monotonic_s,
                            "wrist_seq": wrist_sample.seq,
                            "wrist_monotonic_s": wrist_sample.monotonic_s,
                            "pair_skew_s": abs(
                                front_sample.monotonic_s - wrist_sample.monotonic_s
                            ),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._frames += 1
        except BaseException as exc:  # The owning bridge surfaces this on close/read.
            self._error = exc
        finally:
            self._ready.set()
            if timeline is not None:
                try:
                    timeline.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            for container, stream in (
                (front_container, front_stream),
                (wrist_container, wrist_stream),
            ):
                if container is None:
                    continue
                try:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                    container.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc

    @staticmethod
    def _validated_image(
        value: np.ndarray,
        *,
        width: int,
        height: int,
        label: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.shape != (height, width, 3):
            raise ValueError(
                f"{label} recording frame must have shape "
                f"({height}, {width}, 3), got "
                f"{getattr(value, 'shape', None)}"
            )
        if value.dtype != np.uint8:
            raise ValueError(
                f"{label} recording frame must be uint8, got {value.dtype}"
            )
        return np.ascontiguousarray(value).copy()


def _validate_output_filename(filename: str, *, suffix: str) -> None:
    if (
        not filename
        or filename.startswith(".")
        or Path(filename).name != filename
        or "\\" in filename
        or not filename.endswith(suffix)
        or len(filename.encode("utf-8")) > 240
    ):
        raise ValueError(f"invalid recording output filename: {filename!r}")


def _open_h264_stream(av, path: Path, *, rate: Fraction, width: int, height: int):
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "20"}
    return container, stream
