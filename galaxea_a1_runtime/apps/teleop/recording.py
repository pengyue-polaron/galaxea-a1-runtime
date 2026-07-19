"""Fresh, atomic frame recording for a raw teleoperation episode."""

from __future__ import annotations

import csv
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import rospy

from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    StateMode,
    normalize_episode_decision,
    state_names_for_mode,
    teleop_frame_header,
)
from galaxea_a1_runtime.hardware.cameras import CameraReader, CameraSample
from galaxea_a1_runtime.hardware.image_geometry import crop_image
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES


@dataclass(frozen=True)
class RecordedEpisode:
    frame_count: int
    decision: EpisodeDecision
    actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CapturedFrame:
    row: tuple[Any, ...]
    action: tuple[float, ...]
    camera_seq: dict[str, int]


@dataclass(frozen=True)
class _FrameRecorder:
    episode_dir: Path
    front_reader: CameraReader
    wrist_reader: CameraReader
    ros_state: RosTeleopState
    state_mode: StateMode
    depth_enabled: bool
    front_crop: ImageRoi | None
    max_camera_age_s: float
    max_camera_pair_skew_s: float
    jpeg_params: list[int]

    def capture(
        self, frame_index: int, last_camera_seq: dict[str, int]
    ) -> CapturedFrame | None:
        readers = (self.front_reader, self.wrist_reader)
        _raise_camera_reader_errors(readers)
        wait_for_new_camera_samples(
            readers,
            min_seq=last_camera_seq,
            timeout_s=self.max_camera_age_s,
        )
        now = time.perf_counter()
        front_sample = _fresh_camera_sample(
            self.front_reader, now_s=now, max_age_s=self.max_camera_age_s
        )
        wrist_sample = _fresh_camera_sample(
            self.wrist_reader, now_s=now, max_age_s=self.max_camera_age_s
        )
        camera_skew_s = abs(front_sample.monotonic_s - wrist_sample.monotonic_s)
        if camera_skew_s > self.max_camera_pair_skew_s:
            raise RuntimeError(
                "camera pair is not synchronized: "
                f"skew={camera_skew_s:.3f}s, max={self.max_camera_pair_skew_s:.3f}s, "
                f"front_seq={front_sample.seq}, wrist_seq={wrist_sample.seq}"
            )
        frameset = front_sample.value
        wrist_image = wrist_sample.value
        state_sample = self.ros_state.state_sample(self.state_mode)
        action = self.ros_state.action_values()
        if frameset is None or wrist_image is None:
            raise RuntimeError("camera reader returned an empty frame")
        if state_sample is None or action is None:
            raise RuntimeError(
                "ROS joint/EEF/action/gripper data became stale or invalid while recording"
            )
        if self.depth_enabled and frameset.depth_mm is None:
            return None

        filename = f"{frame_index:06d}.jpg"
        front_image = _crop_if_needed(
            frameset.color_bgr, self.front_crop, label="AgentView color"
        )
        if not cv2.imwrite(
            str(self.episode_dir / "cam0" / filename),
            front_image,
            self.jpeg_params,
        ) or not cv2.imwrite(
            str(self.episode_dir / "cam1" / filename),
            wrist_image,
            self.jpeg_params,
        ):
            raise RuntimeError(f"failed to write camera frame {filename}")
        row: list[Any] = [
            frame_index,
            time.time_ns(),
            f"{state_sample.ros_stamp_s:.9f}",
            front_sample.seq,
            f"{front_sample.monotonic_s:.9f}",
            wrist_sample.seq,
            f"{wrist_sample.monotonic_s:.9f}",
            f"cam0/{filename}",
            f"cam1/{filename}",
        ]
        if self.depth_enabled:
            depth_filename = f"{frame_index:06d}.png"
            depth = _crop_if_needed(
                frameset.depth_mm,
                self.front_crop,
                label="AgentView aligned depth",
            )
            if not cv2.imwrite(
                str(self.episode_dir / "cam0_depth" / depth_filename), depth
            ):
                raise RuntimeError(f"failed to write depth frame {depth_filename}")
            row.append(f"cam0_depth/{depth_filename}")
        return CapturedFrame(
            row=(*row, *state_sample.values, *action),
            action=tuple(float(value) for value in action),
            camera_seq={
                self.front_reader.name: front_sample.seq,
                self.wrist_reader.name: wrist_sample.seq,
            },
        )


def record_episode(
    *,
    episode_dir: Path,
    front_reader: CameraReader,
    wrist_reader: CameraReader,
    ros_state: RosTeleopState,
    state_mode: StateMode,
    fps: float,
    max_duration_s: float,
    jpeg_quality: int,
    depth_enabled: bool,
    front_crop: ImageRoi | None,
    camera_ready_timeout_s: float,
    max_camera_age_s: float,
    max_camera_pair_skew_s: float,
) -> RecordedEpisode:
    (episode_dir / "cam0").mkdir(parents=True)
    (episode_dir / "cam1").mkdir(parents=True)
    if depth_enabled:
        (episode_dir / "cam0_depth").mkdir(parents=True)
    state_names = state_names_for_mode(state_mode)
    action_names = JOINT_ACTION_NAMES
    camera_dirs = ("cam0", "cam1", *(("cam0_depth",) if depth_enabled else ()))
    header = teleop_frame_header(
        state_names=state_names,
        action_names=action_names,
        camera_dirs=camera_dirs,
    )

    frame_index = 0
    wait_for_new_camera_samples(
        (front_reader, wrist_reader),
        min_seq={
            front_reader.name: front_reader.latest_seq(),
            wrist_reader.name: wrist_reader.latest_seq(),
        },
        timeout_s=camera_ready_timeout_s,
    )
    t0 = time.perf_counter()
    next_frame_t = t0
    period = 1.0 / fps
    recorder = _FrameRecorder(
        episode_dir=episode_dir,
        front_reader=front_reader,
        wrist_reader=wrist_reader,
        ros_state=ros_state,
        state_mode=state_mode,
        depth_enabled=depth_enabled,
        front_crop=front_crop,
        max_camera_age_s=max_camera_age_s,
        max_camera_pair_skew_s=max_camera_pair_skew_s,
        jpeg_params=[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    user_input: str | None = None
    last_camera_seq = {front_reader.name: -1, wrist_reader.name: -1}
    actions: list[tuple[float, ...]] = []

    with (episode_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        while not rospy.is_shutdown():
            loop_t = time.perf_counter()
            user_input = _poll_stdin_line()
            if user_input is not None:
                break
            if max_duration_s > 0 and loop_t - t0 >= max_duration_s:
                break

            captured = recorder.capture(frame_index, last_camera_seq)
            if captured is None:
                time.sleep(0.005)
                continue
            writer.writerow(captured.row)
            last_camera_seq = captured.camera_seq
            actions.append(captured.action)
            if frame_index % 30 == 0:
                handle.flush()
            frame_index += 1

            next_frame_t += period
            sleep_s = next_frame_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    return RecordedEpisode(
        frame_count=frame_index,
        decision=normalize_episode_decision(user_input),
        actions=tuple(actions),
    )


def _crop_if_needed(image: Any, roi: ImageRoi | None, *, label: str) -> Any:
    return image if roi is None else crop_image(image, roi, label=label)


def wait_for_new_camera_samples(
    readers: tuple[CameraReader, ...],
    *,
    min_seq: dict[str, int],
    timeout_s: float,
) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        _raise_camera_reader_errors(readers)
        ready = True
        for reader in readers:
            latest = reader.latest()
            if latest is None or latest.seq <= min_seq.get(reader.name, -1):
                ready = False
                break
        if ready:
            return
        time.sleep(0.005)
    details = ", ".join(
        f"{reader.name}:seq={reader.latest_seq()}" for reader in readers
    )
    raise RuntimeError(
        f"camera readers did not produce fresh frames within {timeout_s:.1f}s ({details})"
    )


def _fresh_camera_sample(
    reader: CameraReader, *, now_s: float, max_age_s: float
) -> CameraSample:
    sample = reader.latest()
    if sample is None:
        raise RuntimeError(f"{reader.name} camera has no sample")
    age_s = now_s - sample.monotonic_s
    if age_s > max_age_s:
        raise RuntimeError(
            f"{reader.name} camera sample is stale: age={age_s:.3f}s, "
            f"max={max_age_s:.3f}s, seq={sample.seq}"
        )
    return sample


def _raise_camera_reader_errors(readers: tuple[CameraReader, ...]) -> None:
    for reader in readers:
        exc = reader.exception()
        if exc is not None:
            raise RuntimeError(f"{reader.name} camera reader failed") from exc


def _poll_stdin_line() -> str | None:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    line = sys.stdin.readline()
    if line == "":
        return "q"
    return line.strip().lower()
