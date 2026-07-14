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
from galaxea_a1_runtime.hardware.cameras import CameraSample, LatestCameraReader
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, crop_image
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES


@dataclass(frozen=True)
class RecordedEpisode:
    frame_count: int
    decision: EpisodeDecision
    actions: tuple[tuple[float, ...], ...]


def record_episode(
    *,
    episode_dir: Path,
    front_reader: LatestCameraReader,
    wrist_reader: LatestCameraReader,
    ros_state: RosTeleopState,
    state_mode: StateMode,
    fps: float,
    max_duration_s: float,
    jpeg_quality: int,
    depth_enabled: bool,
    front_crop: ImageRoi | None,
    camera_ready_timeout_s: float,
    max_camera_age_s: float,
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
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
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

            _raise_camera_reader_errors((front_reader, wrist_reader))
            wait_for_new_camera_samples(
                (front_reader, wrist_reader),
                min_seq=last_camera_seq,
                timeout_s=max_camera_age_s,
            )
            sample_t = time.perf_counter()
            front_sample = _fresh_camera_sample(
                front_reader, now_s=sample_t, max_age_s=max_camera_age_s
            )
            wrist_sample = _fresh_camera_sample(
                wrist_reader, now_s=sample_t, max_age_s=max_camera_age_s
            )

            frameset0 = front_sample.value
            img1 = wrist_sample.value
            state = ros_state.state_values(state_mode)
            action = ros_state.action_values()
            if frameset0 is None or img1 is None:
                raise RuntimeError("camera reader returned an empty frame")
            if state is None or action is None:
                raise RuntimeError(
                    "ROS joint/EEF/action/gripper data became stale or invalid while recording"
                )
            if depth_enabled and frameset0.depth_mm is None:
                time.sleep(0.005)
                continue
            last_camera_seq = {
                front_reader.name: front_sample.seq,
                wrist_reader.name: wrist_sample.seq,
            }

            color_filename = f"{frame_index:06d}.jpg"
            front_color = (
                frameset0.color_bgr
                if front_crop is None
                else crop_image(
                    frameset0.color_bgr, front_crop, label="AgentView color"
                )
            )
            ok0 = cv2.imwrite(
                str(episode_dir / "cam0" / color_filename), front_color, jpeg_params
            )
            ok1 = cv2.imwrite(
                str(episode_dir / "cam1" / color_filename), img1, jpeg_params
            )
            if not ok0 or not ok1:
                raise RuntimeError(f"failed to write camera frame {color_filename}")
            row: list[Any] = [
                frame_index,
                time.time_ns(),
                f"{ros_state.ros_stamp():.9f}",
                f"cam0/{color_filename}",
                f"cam1/{color_filename}",
            ]
            if depth_enabled:
                depth_filename = f"{frame_index:06d}.png"
                front_depth = (
                    frameset0.depth_mm
                    if front_crop is None
                    else crop_image(
                        frameset0.depth_mm, front_crop, label="AgentView aligned depth"
                    )
                )
                ok_depth = cv2.imwrite(
                    str(episode_dir / "cam0_depth" / depth_filename), front_depth
                )
                if not ok_depth:
                    raise RuntimeError(f"failed to write depth frame {depth_filename}")
                row.append(f"cam0_depth/{depth_filename}")
            writer.writerow([*row, *state, *action])
            actions.append(tuple(action))
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


def wait_for_new_camera_samples(
    readers: tuple[LatestCameraReader, ...],
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
    reader: LatestCameraReader, *, now_s: float, max_age_s: float
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


def _raise_camera_reader_errors(readers: tuple[LatestCameraReader, ...]) -> None:
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
