#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive multi-episode teleoperation recorder for Galaxea A1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

ROOT_DIR = Path(__file__).resolve().parents[3]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_ROS1_OVERLAY),
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

from galaxea_a1_runtime.collection import (
    CameraMetadata,
    EpisodeDecision,
    StateMode,
    TeleopRawEpisodeMetadata,
    metadata_to_json_dict,
    next_episode_index,
    normalize_episode_decision,
    state_names_for_mode,
    teleop_frame_header,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.hardware.cameras import OpenCVColorCamera, RealSenseColorCamera
from galaxea_a1_runtime.schema import ActionMode, JOINT_ACTION_NAMES


@dataclass(frozen=True)
class JointSnapshot:
    ros_stamp_s: float
    names: tuple[str, ...]
    positions: tuple[float, ...]


class LatestMessageCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._msg: Any | None = None

    def callback(self, msg: Any) -> None:
        with self._lock:
            self._msg = msg

    def get(self) -> Any | None:
        with self._lock:
            return self._msg


class RosTeleopState:
    def __init__(self, args):
        self.args = args
        self.joints = LatestMessageCache()
        self.eef = LatestMessageCache()
        self.action = LatestMessageCache()
        self.gripper_feedback = LatestMessageCache()
        self.gripper_action = LatestMessageCache()

        rospy.Subscriber(args.joint_topic, JointState, self.joints.callback, queue_size=10)
        rospy.Subscriber(args.eef_topic, PoseStamped, self.eef.callback, queue_size=10)
        rospy.Subscriber(args.action_topic, JointState, self.action.callback, queue_size=10)
        rospy.Subscriber(
            args.gripper_feedback_topic,
            JointState,
            self.gripper_feedback.callback,
            queue_size=10,
        )
        rospy.Subscriber(
            args.gripper_action_topic,
            gripper_position_control,
            self.gripper_action.callback,
            queue_size=10,
        )

    def wait_ready(self, *, state_mode: StateMode, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            if self.joint_snapshot() is None:
                time.sleep(0.05)
                continue
            if state_mode in (StateMode.EEF, StateMode.EEF_JOINT) and self.eef_vector() is None:
                time.sleep(0.05)
                continue
            return
        raise RuntimeError(f"ROS state did not become ready within {timeout_s:.1f}s")

    def joint_snapshot(self) -> JointSnapshot | None:
        msg = self.joints.get()
        if msg is None:
            return None
        names = tuple(str(name) for name in getattr(msg, "name", ()))
        positions = tuple(float(value) for value in getattr(msg, "position", ()))
        if not positions:
            return None
        usable = min(len(names), len(positions)) if names else len(positions)
        return JointSnapshot(
            ros_stamp_s=_stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None)),
            names=names[:usable] if names else tuple(f"joint_{i + 1}" for i in range(usable)),
            positions=positions[:usable],
        )

    def eef_vector(self) -> tuple[float, ...] | None:
        msg = self.eef.get()
        if msg is None:
            return None
        pose = msg.pose
        quat = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        norm = float(np.linalg.norm(np.asarray(quat, dtype=np.float64)))
        if norm < 1e-9:
            return None
        quat = tuple(value / norm for value in quat)
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            *quat,
        )

    def state_values(self, mode: StateMode) -> tuple[float, ...] | None:
        joints = self.joint_snapshot()
        if joints is None:
            return None
        joint_values = _first_n(joints.positions, 6, label="joint state")
        gripper = self.gripper_feedback_norm(fallback_joint=joints)
        if gripper is None:
            return None
        eef = self.eef_vector()
        if mode == StateMode.EEF:
            if eef is None:
                return None
            return (*eef, gripper)
        if mode == StateMode.JOINT:
            return (*joint_values, gripper)
        if eef is None:
            return None
        return (*eef, *joint_values, gripper)

    def action_values(self) -> tuple[float, ...] | None:
        msg = self.action.get()
        if msg is None:
            return None
        positions = tuple(float(value) for value in getattr(msg, "position", ()))
        target = _first_n(positions, 6, label="teleop action")
        gripper = self.gripper_action_norm()
        if gripper is None:
            gripper = self.gripper_feedback_norm()
        if gripper is None:
            gripper = 0.0
        return (*target, gripper)

    def gripper_feedback_norm(self, fallback_joint: JointSnapshot | None = None) -> float | None:
        msg = self.gripper_feedback.get()
        if msg is not None and getattr(msg, "position", None):
            return _stroke_to_norm(float(msg.position[0]), self.args.gripper_stroke_scale)
        if fallback_joint is None:
            fallback_joint = self.joint_snapshot()
        if fallback_joint is not None and len(fallback_joint.positions) >= 7:
            return _stroke_to_norm(float(fallback_joint.positions[6]), self.args.gripper_stroke_scale)
        return None

    def gripper_action_norm(self) -> float | None:
        msg = self.gripper_action.get()
        if msg is None:
            return None
        stroke = getattr(msg, "gripper_stroke", None)
        if stroke is None:
            return None
        return _stroke_to_norm(float(stroke), self.args.gripper_stroke_scale)

    def ros_stamp(self) -> float:
        joint = self.joint_snapshot()
        return 0.0 if joint is None else joint.ros_stamp_s


def load_or_prompt_task(experiment_dir: Path, explicit_task: str | None) -> str:
    task_path = experiment_dir / "task.txt"
    if explicit_task:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        task_path.write_text(explicit_task.strip() + "\n")
        return explicit_task.strip()
    if task_path.is_file():
        task = task_path.read_text().strip()
        if task:
            return task
    print("  [first run] Enter task prompt for this experiment:")
    task = input("  > ").strip()
    if not task:
        raise RuntimeError("task prompt cannot be empty")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    task_path.write_text(task + "\n")
    return task


def record_episode(
    *,
    episode_dir: Path,
    front: RealSenseColorCamera,
    wrist: OpenCVColorCamera,
    ros_state: RosTeleopState,
    state_mode: StateMode,
    fps: float,
    max_duration_s: float,
    jpeg_quality: int,
    depth_enabled: bool,
) -> tuple[int, EpisodeDecision]:
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
    t0 = time.perf_counter()
    period = 1.0 / fps
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    user_input: str | None = None

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

            frameset0 = front.read_frameset()
            img1 = wrist.read_bgr()
            state = ros_state.state_values(state_mode)
            action = ros_state.action_values()
            if frameset0 is None or img1 is None or state is None or action is None:
                time.sleep(0.005)
                continue
            if depth_enabled and frameset0.depth_mm is None:
                time.sleep(0.005)
                continue

            color_filename = f"{frame_index:06d}.jpg"
            ok0 = cv2.imwrite(str(episode_dir / "cam0" / color_filename), frameset0.color_bgr, jpeg_params)
            ok1 = cv2.imwrite(str(episode_dir / "cam1" / color_filename), img1, jpeg_params)
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
                ok_depth = cv2.imwrite(str(episode_dir / "cam0_depth" / depth_filename), frameset0.depth_mm)
                if not ok_depth:
                    raise RuntimeError(f"failed to write depth frame {depth_filename}")
                row.append(f"cam0_depth/{depth_filename}")
            writer.writerow([*row, *state, *action])
            if frame_index % 30 == 0:
                handle.flush()
            frame_index += 1

            sleep_s = period - (time.perf_counter() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)

    return frame_index, normalize_episode_decision(user_input)


def write_metadata(
    *,
    episode_dir: Path,
    task: str,
    experiment: str,
    episode_index: int,
    frame_count: int,
    fps: float,
    state_mode: StateMode,
    cam0_serial: str | None,
    cam0_width: int,
    cam0_height: int,
    cam0_depth_enabled: bool,
    cam0_depth_width: int,
    cam0_depth_height: int,
    cam0_depth_aligned: bool,
    cam1_label: str,
    cam1_width: int,
    cam1_height: int,
    args: argparse.Namespace,
) -> None:
    cameras = [
        CameraMetadata("front", "cam0", cam0_width, cam0_height, cam0_serial),
        CameraMetadata("wrist", "cam1", cam1_width, cam1_height, cam1_label),
    ]
    if cam0_depth_enabled:
        cameras.append(
            CameraMetadata(
                "front_depth",
                "cam0_depth",
                cam0_depth_width,
                cam0_depth_height,
                cam0_serial,
                modality="depth",
                dtype="uint16",
                encoding="z16_mm_aligned_to_color" if cam0_depth_aligned else "z16_mm",
            )
        )
    metadata = TeleopRawEpisodeMetadata(
        schema_version=TELEOP_RAW_SCHEMA_VERSION,
        collection_mode="teleop",
        task=task,
        experiment=experiment,
        episode_index=episode_index,
        frame_count=frame_count,
        fps_target=fps,
        state_mode=state_mode,
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=state_names_for_mode(state_mode),
        action_names=JOINT_ACTION_NAMES,
        state_topics={
            "joint": args.joint_topic,
            "eef": args.eef_topic,
            "gripper_feedback": args.gripper_feedback_topic,
        },
        action_topics={
            "joint_target": args.action_topic,
            "gripper_command": args.gripper_action_topic,
        },
        control_path=(
            args.action_topic,
            "jointTracker_demo_node",
            args.staged_command_topic,
            "safe_arm_command_relay_v2.py",
            args.host_command_topic,
        ),
        cameras=tuple(cameras),
    )
    (episode_dir / "metadata.json").write_text(json.dumps(metadata_to_json_dict(metadata), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Galaxea A1 teleop episodes.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--task")
    parser.add_argument("--state-mode", choices=[item.value for item in StateMode], default=StateMode.JOINT.value)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration-s", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--ready-timeout-s", "--joint-wait-timeout-s", dest="ready_timeout_s", type=float, default=10.0)
    parser.add_argument("--joint-topic", default="/joint_states_host")
    parser.add_argument("--eef-topic", default="/end_effector_pose")
    parser.add_argument("--action-topic", default="/arm_joint_target_position")
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument("--gripper-action-topic", default="/gripper_position_control_host")
    parser.add_argument("--gripper-stroke-scale", type=float, default=200.0)
    parser.add_argument("--staged-command-topic", default="/arm_joint_command_a1_staged")
    parser.add_argument("--host-command-topic", default="/arm_joint_command_host")
    parser.add_argument("--cam0-serial")
    parser.add_argument("--cam0-width", type=int, default=640)
    parser.add_argument("--cam0-height", type=int, default=480)
    parser.add_argument("--cam0-fps", type=int, default=30)
    parser.add_argument("--cam0-depth-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cam0-depth-width", type=int, default=640)
    parser.add_argument("--cam0-depth-height", type=int, default=480)
    parser.add_argument("--cam0-align-depth-to-color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam1-device", "--cam1-index", dest="cam1_device", default="auto")
    parser.add_argument("--cam1-width", type=int, default=640)
    parser.add_argument("--cam1-height", type=int, default=480)
    parser.add_argument("--cam1-fps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.cam0_depth_enabled and (args.cam0_depth_width <= 0 or args.cam0_depth_height <= 0):
        raise ValueError("--cam0-depth-width and --cam0-depth-height must be positive when depth is enabled")
    state_mode = StateMode(args.state_mode)
    experiment_dir = args.data_root.expanduser().resolve() / args.experiment
    task = load_or_prompt_task(experiment_dir, args.task)

    rospy.init_node("a1_teleop_collect", anonymous=False, disable_signals=True)
    ros_state = RosTeleopState(args)
    print("[collect] waiting for ROS state ...", end=" ", flush=True)
    ros_state.wait_ready(state_mode=state_mode, timeout_s=args.ready_timeout_s)
    print("ok")

    episode_index = next_episode_index(experiment_dir)
    front: RealSenseColorCamera | None = None
    wrist: OpenCVColorCamera | None = None
    try:
        print("[collect] opening cameras ...", end=" ", flush=True)
        front = RealSenseColorCamera(
            args.cam0_serial,
            args.cam0_width,
            args.cam0_height,
            args.cam0_fps,
            enable_depth=args.cam0_depth_enabled,
            depth_width=args.cam0_depth_width,
            depth_height=args.cam0_depth_height,
            align_depth_to_color=args.cam0_align_depth_to_color,
            warmup_frames=20,
        )
        wrist = OpenCVColorCamera(
            args.cam1_device,
            args.cam1_width,
            args.cam1_height,
            args.cam1_fps,
            warmup_frames=10,
        )
        depth_label = "on" if args.cam0_depth_enabled else "off"
        print(f"ok (wrist={wrist.label}, realsense_depth={depth_label})")

        print(f"\n  experiment  : {args.experiment}")
        print(f"  task        : {task}")
        print(f"  state_mode  : {state_mode.value}")
        print(f"  action_mode : {ActionMode.JOINT_ABSOLUTE.value}")
        print(f"  output      : {experiment_dir}")
        print(f"  next episode: {episode_index}")
        print("  Ctrl+C to quit\n")

        while not rospy.is_shutdown():
            command = input(f"  [{episode_index}] Enter=start recording, q=quit ...").strip().lower()
            if normalize_episode_decision(command) == EpisodeDecision.QUIT:
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{episode_index:03d}_{timestamp}"
            episode_dir = experiment_dir / episode_name
            print(f"  [{episode_index}] recording ... Enter=save, d+Enter=discard, q+Enter=quit")
            frame_count, decision = record_episode(
                episode_dir=episode_dir,
                front=front,
                wrist=wrist,
                ros_state=ros_state,
                state_mode=state_mode,
                fps=args.fps,
                max_duration_s=args.max_duration_s,
                jpeg_quality=args.jpeg_quality,
                depth_enabled=args.cam0_depth_enabled,
            )

            if decision != EpisodeDecision.SAVE or frame_count == 0:
                shutil.rmtree(episode_dir, ignore_errors=True)
                reason = "0 frames" if frame_count == 0 else f"user selected {decision.value}"
                print(f"  [{episode_index}] {reason}; episode deleted.\n")
                if decision == EpisodeDecision.QUIT:
                    break
                continue

            write_metadata(
                episode_dir=episode_dir,
                task=task,
                experiment=args.experiment,
                episode_index=episode_index,
                frame_count=frame_count,
                fps=args.fps,
                state_mode=state_mode,
                cam0_serial=args.cam0_serial,
                cam0_width=args.cam0_width,
                cam0_height=args.cam0_height,
                cam0_depth_enabled=args.cam0_depth_enabled,
                cam0_depth_width=args.cam0_width if args.cam0_align_depth_to_color else args.cam0_depth_width,
                cam0_depth_height=args.cam0_height if args.cam0_align_depth_to_color else args.cam0_depth_height,
                cam0_depth_aligned=args.cam0_align_depth_to_color,
                cam1_label=wrist.label,
                cam1_width=args.cam1_width,
                cam1_height=args.cam1_height,
                args=args,
            )
            print(f"  [{episode_index}] saved {frame_count} frames -> {episode_name}\n")
            episode_index += 1
    except (KeyboardInterrupt, EOFError):
        print(f"\n[collect] done. Next episode index would be {episode_index}.")
    finally:
        if wrist is not None:
            wrist.close()
        if front is not None:
            front.close()
    return 0


def _stamp_to_seconds(stamp: Any) -> float:
    if stamp is None:
        return 0.0
    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        try:
            return float(to_sec())
        except Exception:
            return 0.0
    return float(getattr(stamp, "secs", 0)) + float(getattr(stamp, "nsecs", 0)) * 1e-9


def _first_n(values: tuple[float, ...], count: int, *, label: str) -> tuple[float, ...]:
    if len(values) < count:
        raise RuntimeError(f"{label} has {len(values)} values, need {count}")
    return tuple(float(value) for value in values[:count])


def _stroke_to_norm(stroke_mm: float, scale: float) -> float:
    if scale == 0:
        return 0.0
    return float(np.clip(float(stroke_mm) / float(scale), 0.0, 1.0))


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


if __name__ == "__main__":
    raise SystemExit(main())
