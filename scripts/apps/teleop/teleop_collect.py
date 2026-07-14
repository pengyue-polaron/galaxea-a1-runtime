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
import subprocess
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
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
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
    find_joint_action_step_violation,
    metadata_to_json_dict,
    next_episode_index,
    normalize_episode_decision,
    state_names_for_mode,
    teleop_frame_header,
    validate_existing_camera_shape,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.hardware.cameras import (
    CameraSample,
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    open_color_camera,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, crop_image
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    add_web_preview_arguments,
    color_from_bgr,
    color_from_frameset,
    web_preview_config_from_args,
)
from galaxea_a1_runtime.schema import ActionMode, JOINT_ACTION_NAMES


def styled(text: str, code: str, *, stream: Any = sys.stdout) -> str:
    if not stream.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


@dataclass(frozen=True)
class JointSnapshot:
    ros_stamp_s: float
    names: tuple[str, ...]
    positions: tuple[float, ...]


@dataclass(frozen=True)
class RecordedEpisode:
    frame_count: int
    decision: EpisodeDecision
    actions: tuple[tuple[float, ...], ...]


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
            if self.action_values() is None:
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
            return _binary_gripper(
                _stroke_to_norm(float(msg.position[0]), self.args.gripper_stroke_scale),
                self.args.gripper_binary_open_threshold,
            )
        if fallback_joint is None:
            fallback_joint = self.joint_snapshot()
        if fallback_joint is not None and len(fallback_joint.positions) >= 7:
            return _binary_gripper(
                _stroke_to_norm(float(fallback_joint.positions[6]), self.args.gripper_stroke_scale),
                self.args.gripper_binary_open_threshold,
            )
        return None

    def gripper_action_norm(self) -> float | None:
        msg = self.gripper_action.get()
        if msg is None:
            return None
        stroke = getattr(msg, "gripper_stroke", None)
        if stroke is None:
            return None
        return _binary_gripper(
            _stroke_to_norm(float(stroke), self.args.gripper_stroke_scale),
            self.args.gripper_binary_open_threshold,
        )

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
    print(styled("  First run: enter the task prompt", "1;36"))
    task = input(styled("  > ", "1;36")).strip()
    if not task:
        raise RuntimeError("task prompt cannot be empty")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    task_path.write_text(task + "\n")
    return task


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
    _wait_for_new_camera_samples(
        (front_reader, wrist_reader),
        min_seq={front_reader.name: front_reader.latest_seq(), wrist_reader.name: wrist_reader.latest_seq()},
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
            _wait_for_new_camera_samples(
                (front_reader, wrist_reader),
                min_seq=last_camera_seq,
                timeout_s=max_camera_age_s,
            )
            sample_t = time.perf_counter()
            front_sample = _fresh_camera_sample(front_reader, now_s=sample_t, max_age_s=max_camera_age_s)
            wrist_sample = _fresh_camera_sample(wrist_reader, now_s=sample_t, max_age_s=max_camera_age_s)

            frameset0 = front_sample.value
            img1 = wrist_sample.value
            state = ros_state.state_values(state_mode)
            action = ros_state.action_values()
            if frameset0 is None or img1 is None or state is None or action is None:
                time.sleep(0.005)
                continue
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
                else crop_image(frameset0.color_bgr, front_crop, label="AgentView color")
            )
            ok0 = cv2.imwrite(str(episode_dir / "cam0" / color_filename), front_color, jpeg_params)
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
                front_depth = (
                    frameset0.depth_mm
                    if front_crop is None
                    else crop_image(frameset0.depth_mm, front_crop, label="AgentView aligned depth")
                )
                ok_depth = cv2.imwrite(str(episode_dir / "cam0_depth" / depth_filename), front_depth)
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
    cam0_source_width: int,
    cam0_source_height: int,
    cam0_crop: ImageRoi | None,
    cam1_label: str,
    cam1_width: int,
    cam1_height: int,
    args: argparse.Namespace,
) -> None:
    cameras = [
        CameraMetadata(
            "front",
            "cam0",
            cam0_width,
            cam0_height,
            cam0_serial,
            source_width=cam0_source_width,
            source_height=cam0_source_height,
            crop_xywh=None if cam0_crop is None else cam0_crop.xywh,
        ),
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
                source_width=cam0_source_width,
                source_height=cam0_source_height,
                crop_xywh=None if cam0_crop is None else cam0_crop.xywh,
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
            "safe_arm_command_relay.py",
            args.host_command_topic,
        ),
        cameras=tuple(cameras),
        quality_checks={
            "max_joint_action_step_rad": args.max_joint_action_step_rad,
            "gripper_binary_open_threshold": args.gripper_binary_open_threshold,
        },
    )
    (episode_dir / "metadata.json").write_text(json.dumps(metadata_to_json_dict(metadata), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Galaxea A1 teleop episodes.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--task")
    parser.add_argument(
        "--state-mode",
        choices=[item.value for item in StateMode],
        default=StateMode.EEF_JOINT.value,
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration-s", type=float, default=0.0)
    parser.add_argument("--auto-reset-after-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-runtime-script", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--teleop-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-camera-age-s", type=float, default=0.5)
    parser.add_argument("--max-joint-action-step-rad", type=float, default=0.35)
    parser.add_argument("--joint-topic", default="/joint_states_host")
    parser.add_argument("--eef-topic", default="/end_effector_pose")
    parser.add_argument("--action-topic", default="/arm_joint_target_position")
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument("--gripper-action-topic", default="/gripper_position_control_host")
    parser.add_argument("--gripper-stroke-scale", type=float, default=200.0)
    parser.add_argument("--gripper-binary-open-threshold", type=float, default=0.15)
    parser.add_argument("--staged-command-topic", default="/arm_joint_command_a1_staged")
    parser.add_argument("--host-command-topic", default="/arm_joint_command_host")
    parser.add_argument("--cam0-serial")
    parser.add_argument("--cam0-width", type=int, default=640)
    parser.add_argument("--cam0-height", type=int, default=480)
    parser.add_argument("--cam0-fps", type=int, default=30)
    parser.add_argument("--cam0-require-usb3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam0-depth-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cam0-depth-width", type=int, default=640)
    parser.add_argument("--cam0-depth-height", type=int, default=480)
    parser.add_argument("--cam0-align-depth-to-color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam0-crop-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cam0-crop-x", type=int, default=0)
    parser.add_argument("--cam0-crop-y", type=int, default=0)
    parser.add_argument("--cam0-crop-width", type=int, default=640)
    parser.add_argument("--cam0-crop-height", type=int, default=480)
    parser.add_argument("--cam1-device", default="auto")
    parser.add_argument("--cam1-backend", choices=("realsense", "v4l2"), default="v4l2")
    parser.add_argument("--cam1-serial", default="")
    parser.add_argument("--cam1-width", type=int, default=640)
    parser.add_argument("--cam1-height", type=int, default=480)
    parser.add_argument("--cam1-fps", type=int, default=30)
    parser.add_argument("--cam1-pixel-format", default="YUYV")
    add_web_preview_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_camera_age_s <= 0:
        raise ValueError("--max-camera-age-s must be positive")
    if args.max_joint_action_step_rad <= 0:
        raise ValueError("--max-joint-action-step-rad must be positive")
    if not 0.0 < args.gripper_binary_open_threshold < 1.0:
        raise ValueError("--gripper-binary-open-threshold must be between 0 and 1")
    if args.auto_reset_after_save and (args.reset_runtime_script is None or args.teleop_config is None):
        raise ValueError("automatic reset requires --reset-runtime-script and --teleop-config")
    if args.cam0_depth_enabled and (args.cam0_depth_width <= 0 or args.cam0_depth_height <= 0):
        raise ValueError("--cam0-depth-width and --cam0-depth-height must be positive when depth is enabled")
    front_crop = None
    if args.cam0_crop_enabled:
        front_crop = ImageRoi(
            x=args.cam0_crop_x,
            y=args.cam0_crop_y,
            width=args.cam0_crop_width,
            height=args.cam0_crop_height,
        )
        front_crop.validate(
            image_width=args.cam0_width,
            image_height=args.cam0_height,
            label="AgentView collection ROI",
        )
        if front_crop.width != front_crop.height:
            raise ValueError("AgentView collection ROI must be square")
        if args.cam0_depth_enabled and not args.cam0_align_depth_to_color:
            raise ValueError("AgentView depth must be aligned when collection crop is enabled")
    state_mode = StateMode(args.state_mode)
    experiment_dir = args.data_root.expanduser().resolve() / args.experiment
    task = load_or_prompt_task(experiment_dir, args.task)
    validate_existing_camera_shape(
        experiment_dir,
        camera_name="front",
        width=args.cam0_width if front_crop is None else front_crop.width,
        height=args.cam0_height if front_crop is None else front_crop.height,
    )

    rospy.init_node("a1_teleop_collect", anonymous=False, disable_signals=True)
    ros_state = RosTeleopState(args)
    print(styled("[Setup] ROS state", "1;36"), end=" ... ", flush=True)
    ros_state.wait_ready(state_mode=state_mode, timeout_s=args.ready_timeout_s)
    print(styled("ready", "1;32"))

    episode_index = next_episode_index(experiment_dir)
    front: RealSenseColorCamera | None = None
    wrist: ColorCamera | None = None
    front_reader: LatestCameraReader | None = None
    wrist_reader: LatestCameraReader | None = None
    web_preview: CameraWebPreview | None = None
    try:
        print(styled("[Setup] Cameras", "1;36"), end=" ... ", flush=True)
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
            require_usb3=args.cam0_require_usb3,
        )
        wrist = open_color_camera(
            args.cam1_backend,
            serial=args.cam1_serial,
            device=args.cam1_device,
            width=args.cam1_width,
            height=args.cam1_height,
            fps=args.cam1_fps,
            pixel_format=args.cam1_pixel_format,
            warmup_frames=10,
        )
        front_reader = LatestCameraReader(
            "front",
            front.read_frameset,
        )
        wrist_reader = LatestCameraReader("wrist", wrist.read_bgr)
        front_reader.start()
        wrist_reader.start()
        preview_config = web_preview_config_from_args(args)
        if preview_config.enabled:
            web_preview = CameraWebPreview(preview_config)
            web_preview.register_reader(
                "agent",
                front_reader,
                extract=color_from_frameset,
                source=front.label,
                overlay_roi=front_crop,
                overlay_label=(
                    f"RECORDED {front_crop.width}x{front_crop.height}"
                    if front_crop is not None
                    else ""
                ),
            )
            web_preview.register_reader("wrist", wrist_reader, extract=color_from_bgr, source=wrist.label)
            web_preview.start()
        _wait_for_new_camera_samples(
            (front_reader, wrist_reader),
            min_seq={"front": -1, "wrist": -1},
            timeout_s=args.ready_timeout_s,
        )
        depth_label = "on" if args.cam0_depth_enabled else "off"
        print(
            styled("ready", "1;32")
            + f" (wrist={wrist.label}, realsense_usb={front.usb_type}, depth={depth_label})"
        )

        print(f"\n  experiment  : {args.experiment}")
        print(f"  task        : {task}")
        print(f"  state_mode  : {state_mode.value}")
        print(f"  action_mode : {ActionMode.JOINT_ABSOLUTE.value}")
        print(f"  output      : {experiment_dir}")
        print(f"  AgentView ROI: {'full frame' if front_crop is None else front_crop.xywh}")
        print(f"  next episode: {episode_index}")
        print("  Ctrl+C to quit\n")

        while not rospy.is_shutdown():
            command = input(
                styled(f"  [{episode_index}] Enter=start recording, q=quit > ", "1;36")
            ).strip().lower()
            if normalize_episode_decision(command) == EpisodeDecision.QUIT:
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{episode_index:03d}_{timestamp}"
            episode_dir = experiment_dir / episode_name
            print(
                styled(
                    f"  [{episode_index}] RECORDING",
                    "1;33",
                )
                + "  Enter=save, d+Enter=discard, q+Enter=quit"
            )
            try:
                recording = record_episode(
                    episode_dir=episode_dir,
                    front_reader=front_reader,
                    wrist_reader=wrist_reader,
                    ros_state=ros_state,
                    state_mode=state_mode,
                    fps=args.fps,
                    max_duration_s=args.max_duration_s,
                    jpeg_quality=args.jpeg_quality,
                    depth_enabled=args.cam0_depth_enabled,
                    front_crop=front_crop,
                    camera_ready_timeout_s=args.ready_timeout_s,
                    max_camera_age_s=args.max_camera_age_s,
                )
            except BaseException:
                shutil.rmtree(episode_dir, ignore_errors=True)
                print(
                    styled(
                        f"  [{episode_index}] ERROR: recording failed; episode deleted -> {episode_name}",
                        "1;31",
                        stream=sys.stderr,
                    ),
                    file=sys.stderr,
                )
                raise

            if recording.decision != EpisodeDecision.SAVE or recording.frame_count == 0:
                shutil.rmtree(episode_dir, ignore_errors=True)
                reason = (
                    "0 frames"
                    if recording.frame_count == 0
                    else f"user selected {recording.decision.value}"
                )
                print(f"  [{episode_index}] {reason}; episode deleted.\n")
                if recording.decision == EpisodeDecision.QUIT:
                    break
                continue

            violation = find_joint_action_step_violation(
                recording.actions,
                action_names=JOINT_ACTION_NAMES,
                max_step_rad=args.max_joint_action_step_rad,
            )
            if violation is not None:
                shutil.rmtree(episode_dir, ignore_errors=True)
                print(
                    styled(
                        f"  [{episode_index}] REJECTED: joint action discontinuity: "
                        f"{violation.describe()}",
                        "1;31",
                        stream=sys.stderr,
                    ),
                    file=sys.stderr,
                )
                print(f"  [{episode_index}] episode deleted; index will be reused.\n", file=sys.stderr)
                if args.auto_reset_after_save:
                    reset_for_next_episode(
                        runtime_script=args.reset_runtime_script,
                        teleop_config=args.teleop_config,
                    )
                continue

            write_metadata(
                episode_dir=episode_dir,
                task=task,
                experiment=args.experiment,
                episode_index=episode_index,
                frame_count=recording.frame_count,
                fps=args.fps,
                state_mode=state_mode,
                cam0_serial=args.cam0_serial,
                cam0_width=args.cam0_width if front_crop is None else front_crop.width,
                cam0_height=args.cam0_height if front_crop is None else front_crop.height,
                cam0_depth_enabled=args.cam0_depth_enabled,
                cam0_depth_width=(
                    front_crop.width
                    if front_crop is not None
                    else (args.cam0_width if args.cam0_align_depth_to_color else args.cam0_depth_width)
                ),
                cam0_depth_height=(
                    front_crop.height
                    if front_crop is not None
                    else (args.cam0_height if args.cam0_align_depth_to_color else args.cam0_depth_height)
                ),
                cam0_depth_aligned=args.cam0_align_depth_to_color,
                cam0_source_width=args.cam0_width,
                cam0_source_height=args.cam0_height,
                cam0_crop=front_crop,
                cam1_label=wrist.label,
                cam1_width=args.cam1_width,
                cam1_height=args.cam1_height,
                args=args,
            )
            nominal_s = recording.frame_count / args.fps if args.fps > 0 else 0.0
            print(
                styled(
                    f"  [{episode_index}] SAVED {recording.frame_count} frames "
                    f"(~{nominal_s:.1f}s @ {args.fps:g}fps) -> {episode_name}",
                    "1;32",
                )
                + "\n"
            )
            episode_index += 1
            if args.auto_reset_after_save:
                reset_for_next_episode(
                    runtime_script=args.reset_runtime_script,
                    teleop_config=args.teleop_config,
                )
    except (KeyboardInterrupt, EOFError):
        print(f"\n[collect] done. Next episode index would be {episode_index}.")
    finally:
        if web_preview is not None:
            web_preview.close()
        if wrist_reader is not None:
            wrist_reader.stop()
        if front_reader is not None:
            front_reader.stop()
        if wrist is not None:
            wrist.close()
        if front is not None:
            front.close()
    return 0


def reset_for_next_episode(*, runtime_script: Path, teleop_config: Path) -> None:
    try:
        subprocess.run(
            [
                str(runtime_script.resolve()),
                "--config",
                str(teleop_config.resolve()),
                "_reset-live",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "automatic reset failed; collection stopped before the next episode"
        ) from exc
    print()


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


def _wait_for_new_camera_samples(
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
    details = ", ".join(f"{reader.name}:seq={reader.latest_seq()}" for reader in readers)
    raise RuntimeError(f"camera readers did not produce fresh frames within {timeout_s:.1f}s ({details})")


def _fresh_camera_sample(reader: LatestCameraReader, *, now_s: float, max_age_s: float) -> CameraSample:
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


def _first_n(values: tuple[float, ...], count: int, *, label: str) -> tuple[float, ...]:
    if len(values) < count:
        raise RuntimeError(f"{label} has {len(values)} values, need {count}")
    return tuple(float(value) for value in values[:count])


def _stroke_to_norm(stroke_mm: float, scale: float) -> float:
    if scale == 0:
        return 0.0
    return float(np.clip(float(stroke_mm) / float(scale), 0.0, 1.0))


def _binary_gripper(value: float, open_threshold: float) -> float:
    return 1.0 if value >= open_threshold else 0.0


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
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(styled(f"error: {exc}", "1;31", stream=sys.stderr), file=sys.stderr)
        raise SystemExit(1)
