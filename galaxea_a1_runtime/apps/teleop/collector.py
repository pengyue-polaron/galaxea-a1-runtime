#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive multi-episode teleoperation recorder implementation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR)

import rospy

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.collector_episode import (
    TeleopEpisodeSession,
    styled,
)
from galaxea_a1_runtime.apps.teleop.collector_setup import validate_collector_args
from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    StateMode,
    next_episode_index,
    normalize_episode_decision,
    validate_existing_camera_shape,
    validate_existing_schema,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.hardware.web_preview import add_web_preview_arguments
from galaxea_a1_runtime.schema import ActionMode


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
    state_mode, front_crop = validate_collector_args(args)
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
    cameras = TeleopCameraSession(args, front_crop)
    try:
        print(styled("[Setup] Cameras", "1;36"), end=" ... ", flush=True)
        camera_summary = cameras.start()
        print(styled("ready", "1;32") + f" ({camera_summary})")
        _print_collection_summary(
            args=args,
            task=task,
            state_mode=state_mode,
            experiment_dir=experiment_dir,
            front_crop=front_crop,
            episode_index=episode_index,
        )
        episodes = TeleopEpisodeSession(
            args=args,
            experiment_dir=experiment_dir,
            task=task,
            state_mode=state_mode,
            front_crop=front_crop,
            ros_state=ros_state,
            cameras=cameras,
            repo_root=ROOT_DIR,
        )
        while not rospy.is_shutdown():
            command = (
                input(
                    styled(
                        f"  [{episode_index}] Enter=start recording, q=quit > ",
                        "1;36",
                    )
                )
                .strip()
                .lower()
            )
            if normalize_episode_decision(command) == EpisodeDecision.QUIT:
                break
            completion = episodes.record(episode_index)
            if completion.decision == EpisodeDecision.QUIT:
                break
            if completion.decision == EpisodeDecision.SAVE:
                episode_index += 1
            if completion.reset_required:
                reset_for_next_episode(
                    runtime_script=args.reset_runtime_script,
                    teleop_config=args.teleop_config,
                )
    except (KeyboardInterrupt, EOFError):
        print(f"\n[collect] done. Next episode index would be {episode_index}.")
    finally:
        cameras.close()
    return 0


def _print_collection_summary(
    *,
    args,
    task: str,
    state_mode,
    experiment_dir: Path,
    front_crop,
    episode_index: int,
) -> None:
    print(f"\n  experiment  : {args.experiment}")
    print(f"  task        : {task}")
    print(f"  state_mode  : {state_mode.value}")
    print(f"  action_mode : {ActionMode.JOINT_ABSOLUTE.value}")
    print(f"  output      : {experiment_dir}")
    print(f"  AgentView ROI: {'full frame' if front_crop is None else front_crop.xywh}")
    print(f"  next episode: {episode_index}")
    print("  Ctrl+C to quit\n")


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


def cli_main() -> int:
    try:
        return main()
    except RuntimeError as exc:
        print(styled(f"error: {exc}", "1;31", stream=sys.stderr), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
