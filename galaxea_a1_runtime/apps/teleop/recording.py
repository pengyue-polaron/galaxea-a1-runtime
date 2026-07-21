"""Fresh frame capture into a pending direct LeRobot episode."""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from typing import Any

import rospy
from embodied_ops.collection import require_fresh_sample, require_pair_skew

from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    normalize_episode_decision,
)
from galaxea_a1_runtime.collection.lerobot_frame import build_lerobot_frame
from galaxea_a1_runtime.hardware.cameras import CameraReader, CameraSample
from galaxea_a1_runtime.hardware.image_geometry import crop_image
from galaxea_a1_runtime.configuration.image import ImageRoi


@dataclass(frozen=True)
class RecordedEpisode:
    frame_count: int
    decision: EpisodeDecision
    actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CapturedFrame:
    values: dict[str, Any]
    action: tuple[float, ...]
    camera_seq: dict[str, int]


@dataclass(frozen=True)
class _FrameRecorder:
    front_reader: CameraReader
    wrist_reader: CameraReader
    ros_state: RosTeleopState
    task: str
    depth_enabled: bool
    front_crop: ImageRoi | None
    max_camera_age_s: float
    max_camera_pair_skew_s: float

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
        require_pair_skew(
            front_sample,
            wrist_sample,
            left_label="front",
            right_label="wrist",
            max_skew_s=self.max_camera_pair_skew_s,
        )
        frameset = front_sample.value
        wrist_image = wrist_sample.value
        state_sample = self.ros_state.state_sample()
        action = self.ros_state.action_values()
        if frameset is None or wrist_image is None:
            raise RuntimeError("camera reader returned an empty frame")
        if state_sample is None or action is None:
            raise RuntimeError(
                "ROS joint/EEF/action/gripper data became stale or invalid while recording"
            )
        if self.depth_enabled and frameset.depth_mm is None:
            return None

        front_image = _crop_if_needed(
            frameset.color_bgr, self.front_crop, label="AgentView color"
        )
        depth = None
        if self.depth_enabled:
            depth = _crop_if_needed(
                frameset.depth_mm,
                self.front_crop,
                label="AgentView aligned depth",
            )
        values = build_lerobot_frame(
            state=state_sample.values,
            action=action,
            front_bgr=front_image,
            wrist_bgr=wrist_image,
            task=self.task,
            front_depth_mm=depth,
        )
        return CapturedFrame(
            values=values,
            action=tuple(float(value) for value in action),
            camera_seq={
                self.front_reader.name: front_sample.seq,
                self.wrist_reader.name: wrist_sample.seq,
            },
        )


def record_episode(
    *,
    dataset: Any,
    task: str,
    front_reader: CameraReader,
    wrist_reader: CameraReader,
    ros_state: RosTeleopState,
    fps: float,
    max_duration_s: float,
    depth_enabled: bool,
    front_crop: ImageRoi | None,
    camera_ready_timeout_s: float,
    max_camera_age_s: float,
    max_camera_pair_skew_s: float,
) -> RecordedEpisode:
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
        front_reader=front_reader,
        wrist_reader=wrist_reader,
        ros_state=ros_state,
        task=task,
        depth_enabled=depth_enabled,
        front_crop=front_crop,
        max_camera_age_s=max_camera_age_s,
        max_camera_pair_skew_s=max_camera_pair_skew_s,
    )
    user_input: str | None = None
    last_camera_seq = {front_reader.name: -1, wrist_reader.name: -1}
    actions: list[tuple[float, ...]] = []

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
        dataset.add_frame(captured.values)
        last_camera_seq = captured.camera_seq
        actions.append(captured.action)
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
    return require_fresh_sample(
        reader.latest(),
        label=f"{reader.name} camera",
        now_s=now_s,
        max_age_s=max_age_s,
    )


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
