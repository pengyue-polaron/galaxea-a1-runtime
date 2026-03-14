#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Record one A1 teleoperation episode:
  - cam0: RealSense (agent view)
  - cam1: OpenCV camera (wrist view)
  - /joint_states_host joint angles from A1

The recording runs until the operator presses Enter, then saves the episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow using rospy in conda python by appending common ROS Python paths.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/home/eric/A1_SDK/install/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)


@dataclass
class JointSnapshot:
    ros_stamp_s: float
    names: list[str]
    positions: list[float]


class JointStateCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._msg: Any | None = None

    def callback(self, msg: Any) -> None:
        with self._lock:
            self._msg = msg

    def snapshot(self) -> JointSnapshot | None:
        with self._lock:
            msg = self._msg

        if msg is None:
            return None

        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        if len(names) == 0 or len(positions) == 0:
            return None

        n = min(len(names), len(positions))
        ros_stamp_s = _extract_ros_stamp_seconds(msg)
        return JointSnapshot(ros_stamp_s=ros_stamp_s, names=names[:n], positions=positions[:n])


def _extract_ros_stamp_seconds(msg: Any) -> float:
    header = getattr(msg, "header", None)
    if header is None:
        return 0.0

    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0.0

    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        try:
            return float(to_sec())
        except Exception:
            return 0.0

    secs = getattr(stamp, "secs", getattr(stamp, "sec", 0))
    nsecs = getattr(stamp, "nsecs", getattr(stamp, "nanosec", 0))
    return float(secs) + float(nsecs) / 1_000_000_000.0


def _resolve_conda_rospy() -> tuple[Any, Any]:
    try:
        import rospy
        from sensor_msgs.msg import JointState
    except Exception as exc:
        raise RuntimeError(
            "Failed to import rospy/sensor_msgs. Ensure ROS (Noetic) python packages are installed and accessible."
        ) from exc
    return rospy, JointState


def _open_realsense(
    serial: str | None, width: int, height: int, fps: int
) -> tuple[Any, Any]:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import pyrealsense2. Install RealSense dependencies in the active environment."
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return rs, pipeline


def _open_cam1(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open cam1 index={index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _video_device_name(index: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{index}/name")
    if not name_path.exists():
        return "unknown"
    try:
        return name_path.read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


def _probe_cam_indices(max_index: int = 12) -> list[tuple[int, str]]:
    readable: list[tuple[int, str]] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            readable.append((idx, _video_device_name(idx)))
    return readable


def _open_cam1_with_auto(index_or_auto: str, width: int, height: int, fps: int) -> tuple[cv2.VideoCapture, int]:
    text = str(index_or_auto).strip().lower()

    if text != "auto":
        idx = int(text)
        cap = _open_cam1(idx, width, height, fps)
        return cap, idx

    readable = _probe_cam_indices(max_index=12)
    if len(readable) == 0:
        raise RuntimeError("Failed to auto-detect cam1: no readable OpenCV camera index found in [0, 12].")

    # Prefer a non-RealSense camera for wrist view when available.
    for idx, name in readable:
        lowered = name.lower()
        if "realsense" not in lowered and "intel" not in lowered:
            return _open_cam1(idx, width, height, fps), idx

    idx = readable[0][0]
    return _open_cam1(idx, width, height, fps), idx


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record A1 teleop episode with cam0/cam1 and joint angles.")
    parser.add_argument("--output-root", default="/home/eric/lerobot/data/a1", help="Episode root directory.")
    parser.add_argument("--episode-name", default=None, help="Episode folder name. Default: episode_YYYYmmdd_HHMMSS")
    parser.add_argument("--task", default="A1 single-arm teleop collection", help="Task description.")

    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS.")
    parser.add_argument("--max-duration-s", type=float, default=0.0, help="Optional hard stop duration. 0 disables.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality [0,100].")

    parser.add_argument("--joint-topic", default="/joint_states_host", help="ROS JointState topic for A1.")
    parser.add_argument("--joint-wait-timeout-s", type=float, default=10.0, help="Timeout waiting for first joint state.")

    parser.add_argument("--cam0-serial", default=None, help="Optional RealSense serial for cam0.")
    parser.add_argument("--cam0-width", type=int, default=640)
    parser.add_argument("--cam0-height", type=int, default=480)
    parser.add_argument("--cam0-fps", type=int, default=30)

    parser.add_argument("--cam1-index", default="auto", help="OpenCV index for wrist camera, or 'auto'.")
    parser.add_argument("--cam1-width", type=int, default=640)
    parser.add_argument("--cam1-height", type=int, default=480)
    parser.add_argument("--cam1-fps", type=int, default=30)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [0,100]")

    episode_name = args.episode_name
    if episode_name is None:
        episode_name = datetime.now().strftime("episode_%Y%m%d_%H%M%S")

    episode_dir = Path(args.output_root).expanduser().resolve() / episode_name
    cam0_dir = episode_dir / "cam0"
    cam1_dir = episode_dir / "cam1"
    episode_dir.mkdir(parents=True, exist_ok=False)
    cam0_dir.mkdir(parents=True, exist_ok=False)
    cam1_dir.mkdir(parents=True, exist_ok=False)

    frames_csv_path = episode_dir / "frames.csv"
    metadata_json_path = episode_dir / "metadata.json"

    rospy, JointState = _resolve_conda_rospy()
    rospy.init_node("lerobot_a1_collect", anonymous=False, disable_signals=True)
    joint_cache = JointStateCache()
    _ = rospy.Subscriber(args.joint_topic, JointState, joint_cache.callback, queue_size=10)

    print(f"[collect] output: {episode_dir}")
    print(f"[collect] waiting for first joint state on {args.joint_topic} ...")
    wait_deadline = time.time() + args.joint_wait_timeout_s
    while joint_cache.snapshot() is None:
        if time.time() > wait_deadline:
            raise TimeoutError(
                f"No joint state received from {args.joint_topic} within {args.joint_wait_timeout_s:.1f}s."
            )
        time.sleep(0.05)

    rs, rs_pipeline = _open_realsense(args.cam0_serial, args.cam0_width, args.cam0_height, args.cam0_fps)
    cam1, cam1_index = _open_cam1_with_auto(args.cam1_index, args.cam1_width, args.cam1_height, args.cam1_fps)
    print(f"[collect] cam1 index selected: {cam1_index} ({_video_device_name(cam1_index)})")

    stop_event = threading.Event()

    def _wait_for_enter() -> None:
        try:
            input("[collect] recording... press Enter to finish and save this episode.\n")
        except EOFError:
            pass
        stop_event.set()

    waiter = threading.Thread(target=_wait_for_enter, daemon=True)
    waiter.start()

    frame_idx = 0
    joint_names: list[str] | None = None
    start_iso = datetime.now().isoformat(timespec="seconds")
    start_t = time.perf_counter()
    target_period_s = 1.0 / float(args.fps)
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    try:
        with frames_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            while not stop_event.is_set():
                loop_t = time.perf_counter()

                if args.max_duration_s > 0 and (loop_t - start_t) >= args.max_duration_s:
                    print(f"[collect] reached max duration {args.max_duration_s:.1f}s, stopping.")
                    break

                frames = rs_pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                cam0_img = np.asanyarray(color_frame.get_data())

                ok, cam1_img = cam1.read()
                if not ok:
                    continue

                snapshot = joint_cache.snapshot()
                if snapshot is None:
                    continue

                if joint_names is None:
                    joint_names = snapshot.names
                    writer.writerow(
                        ["frame_index", "wall_time_ns", "ros_stamp_s", "cam0_relpath", "cam1_relpath", *joint_names]
                    )

                pos_map = {name: pos for name, pos in zip(snapshot.names, snapshot.positions)}
                row_positions = [pos_map.get(name, "") for name in joint_names]

                frame_name = f"{frame_idx:06d}.jpg"
                cam0_rel = Path("cam0") / frame_name
                cam1_rel = Path("cam1") / frame_name
                cam0_path = episode_dir / cam0_rel
                cam1_path = episode_dir / cam1_rel

                ok0 = cv2.imwrite(str(cam0_path), cam0_img, jpeg_params)
                ok1 = cv2.imwrite(str(cam1_path), cam1_img, jpeg_params)
                if not ok0 or not ok1:
                    raise RuntimeError("Failed to write camera frame(s) to disk.")

                writer.writerow(
                    [
                        frame_idx,
                        time.time_ns(),
                        f"{snapshot.ros_stamp_s:.9f}",
                        cam0_rel.as_posix(),
                        cam1_rel.as_posix(),
                        *row_positions,
                    ]
                )
                if frame_idx % 30 == 0:
                    f.flush()

                frame_idx += 1

                elapsed = time.perf_counter() - loop_t
                sleep_s = target_period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[collect] interrupted by user, finalizing current episode...")
    finally:
        cam1.release()
        rs_pipeline.stop()

    end_iso = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "task": args.task,
        "episode_dir": str(episode_dir),
        "frame_count": frame_idx,
        "start_time": start_iso,
        "end_time": end_iso,
        "fps_target": args.fps,
        "joint_topic": args.joint_topic,
        "joint_names": joint_names or [],
        "cam0": {
            "type": "intelrealsense",
            "serial": args.cam0_serial,
            "width": args.cam0_width,
            "height": args.cam0_height,
            "fps": args.cam0_fps,
        },
        "cam1": {
            "type": "opencv",
            "index": cam1_index,
            "width": args.cam1_width,
            "height": args.cam1_height,
            "fps": args.cam1_fps,
        },
    }
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[collect] saved {frame_idx} frames.")
    print(f"[collect] frames csv: {frames_csv_path}")
    print(f"[collect] metadata: {metadata_json_path}")


if __name__ == "__main__":
    main()
