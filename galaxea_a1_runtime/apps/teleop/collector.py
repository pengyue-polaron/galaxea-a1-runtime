#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive multi-episode teleoperation recorder implementation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR)

import rospy

from galaxea_a1_runtime.apps.teleop.metadata import write_metadata
from galaxea_a1_runtime.apps.teleop.recording import (
    record_episode,
    wait_for_new_camera_samples,
)
from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    StateMode,
    find_joint_action_step_violation,
    next_episode_index,
    normalize_episode_decision,
    validate_existing_camera_shape,
    validate_existing_schema,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    open_color_camera,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
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
    parser.add_argument(
        "--auto-reset-after-save", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reset-runtime-script", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--teleop-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-camera-age-s", type=float, default=0.5)
    parser.add_argument("--max-joint-feedback-age-s", type=float, required=True)
    parser.add_argument("--max-eef-feedback-age-s", type=float, required=True)
    parser.add_argument("--max-action-age-s", type=float, required=True)
    parser.add_argument("--max-gripper-age-s", type=float, default=0.5)
    parser.add_argument("--max-joint-action-step-rad", type=float, default=0.35)
    parser.add_argument("--joint-topic", default="/joint_states_host")
    parser.add_argument("--eef-topic", default="/end_effector_pose")
    parser.add_argument("--action-topic", default="/arm_joint_target_position")
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument("--gripper-action-topic", required=True)
    parser.add_argument("--gripper-stroke-min", type=float, required=True)
    parser.add_argument("--gripper-stroke-max", type=float, required=True)
    parser.add_argument(
        "--staged-command-topic", default="/arm_joint_command_a1_staged"
    )
    parser.add_argument("--host-command-topic", default="/arm_joint_command_host")
    parser.add_argument("--cam0-serial")
    parser.add_argument("--cam0-width", type=int, default=640)
    parser.add_argument("--cam0-height", type=int, default=480)
    parser.add_argument("--cam0-fps", type=int, default=30)
    parser.add_argument(
        "--cam0-require-usb3", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cam0-depth-enabled", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cam0-depth-width", type=int, default=640)
    parser.add_argument("--cam0-depth-height", type=int, default=480)
    parser.add_argument(
        "--cam0-align-depth-to-color",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cam0-crop-enabled", action=argparse.BooleanOptionalAction, default=False
    )
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
    if args.max_joint_feedback_age_s <= 0:
        raise ValueError("--max-joint-feedback-age-s must be positive")
    if args.max_eef_feedback_age_s <= 0:
        raise ValueError("--max-eef-feedback-age-s must be positive")
    if args.max_action_age_s <= 0:
        raise ValueError("--max-action-age-s must be positive")
    if args.max_gripper_age_s <= 0:
        raise ValueError("--max-gripper-age-s must be positive")
    if args.max_joint_action_step_rad <= 0:
        raise ValueError("--max-joint-action-step-rad must be positive")
    if args.gripper_stroke_max <= args.gripper_stroke_min:
        raise ValueError(
            "--gripper-stroke-max must be greater than --gripper-stroke-min"
        )
    if args.auto_reset_after_save and (
        args.reset_runtime_script is None or args.teleop_config is None
    ):
        raise ValueError(
            "automatic reset requires --reset-runtime-script and --teleop-config"
        )
    if args.cam0_depth_enabled and (
        args.cam0_depth_width <= 0 or args.cam0_depth_height <= 0
    ):
        raise ValueError(
            "--cam0-depth-width and --cam0-depth-height must be positive when depth is enabled"
        )
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
            raise ValueError(
                "AgentView depth must be aligned when collection crop is enabled"
            )
    state_mode = StateMode(args.state_mode)
    experiment_dir = args.data_root.expanduser().resolve() / args.experiment
    task = load_or_prompt_task(experiment_dir, args.task)
    validate_existing_schema(experiment_dir, expected=TELEOP_RAW_SCHEMA_VERSION)
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
            web_preview.register_reader(
                "wrist", wrist_reader, extract=color_from_bgr, source=wrist.label
            )
            web_preview.start()
        wait_for_new_camera_samples(
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
        print(
            f"  AgentView ROI: {'full frame' if front_crop is None else front_crop.xywh}"
        )
        print(f"  next episode: {episode_index}")
        print("  Ctrl+C to quit\n")

        while not rospy.is_shutdown():
            command = (
                input(
                    styled(
                        f"  [{episode_index}] Enter=start recording, q=quit > ", "1;36"
                    )
                )
                .strip()
                .lower()
            )
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
                print(
                    f"  [{episode_index}] episode deleted; index will be reused.\n",
                    file=sys.stderr,
                )
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
                cam0_height=args.cam0_height
                if front_crop is None
                else front_crop.height,
                cam0_depth_enabled=args.cam0_depth_enabled,
                cam0_depth_width=(
                    front_crop.width
                    if front_crop is not None
                    else (
                        args.cam0_width
                        if args.cam0_align_depth_to_color
                        else args.cam0_depth_width
                    )
                ),
                cam0_depth_height=(
                    front_crop.height
                    if front_crop is not None
                    else (
                        args.cam0_height
                        if args.cam0_align_depth_to_color
                        else args.cam0_depth_height
                    )
                ),
                cam0_depth_aligned=args.cam0_align_depth_to_color,
                cam0_source_width=args.cam0_width,
                cam0_source_height=args.cam0_height,
                cam0_crop=front_crop,
                cam1_label=wrist.label,
                cam1_width=args.cam1_width,
                cam1_height=args.cam1_height,
                config_path=_config_reference(args.teleop_config),
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


def _config_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def cli_main() -> int:
    try:
        return main()
    except RuntimeError as exc:
        print(styled(f"error: {exc}", "1;31", stream=sys.stderr), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
