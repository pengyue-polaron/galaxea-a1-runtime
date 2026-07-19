#!/usr/bin/env python3
# ruff: noqa: E402
"""Interactive multi-episode teleoperation recorder implementation."""

from __future__ import annotations

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
from galaxea_a1_runtime.apps.teleop.collector_episode import TeleopEpisodeSession
from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.filesystem import atomic_write_text
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    next_episode_index,
    normalize_episode_decision,
    validate_existing_camera_shape,
    validate_episode_layout,
    validate_experiment_name,
    validate_existing_schema,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.console import Tone, failure, info, step, style, success
from galaxea_a1_runtime.schema import ActionMode
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def load_or_prompt_task(experiment_dir: Path) -> str:
    task_path = experiment_dir / "task.txt"
    if task_path.is_file():
        task = task_path.read_text().strip()
        if task:
            return task
    info("First run: enter the task prompt.")
    task = input(style("Task > ", Tone.STEP)).strip()
    if not task:
        raise RuntimeError("task prompt cannot be empty")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(task_path, task + "\n")
    return task


def run(config: TeleopConfig, *, experiment: str) -> int:
    experiment = validate_experiment_name(experiment)
    state_mode = config.collection.state_mode
    front_crop = required_front_roi(config.system.cameras)
    front = config.system.cameras.front
    experiment_dir = config.collection.data_root / experiment
    validate_episode_layout(experiment_dir)
    validate_existing_schema(experiment_dir, expected=TELEOP_RAW_SCHEMA_VERSION)
    validate_existing_camera_shape(
        experiment_dir,
        camera_name="front",
        width=front.width if front_crop is None else front_crop.width,
        height=front.height if front_crop is None else front_crop.height,
    )
    task = load_or_prompt_task(experiment_dir)

    rospy.init_node("a1_teleop_collect", anonymous=False, disable_signals=True)
    ros_state = RosTeleopState(config)
    step("Waiting for ROS state")
    ros_state.wait_ready(
        state_mode=state_mode, timeout_s=config.collection.ready_timeout_s
    )
    success("ROS state ready.")

    episode_index = next_episode_index(experiment_dir)
    cameras = TeleopCameraSession(config)
    try:
        step("Starting cameras")
        camera_summary = cameras.start()
        success(f"Cameras ready: {camera_summary}")
        _print_collection_summary(
            experiment=experiment,
            task=task,
            state_mode=state_mode,
            experiment_dir=experiment_dir,
            front_crop=front_crop,
            episode_index=episode_index,
        )
        episodes = TeleopEpisodeSession(
            config=config,
            experiment=experiment,
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
                    style(
                        f"  [{episode_index}] Enter=start recording, q=quit > ",
                        Tone.STEP,
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
                reset_for_next_episode(config.path)
    except (KeyboardInterrupt, EOFError):
        print()
        info(f"Collection stopped. Next episode index: {episode_index}.")
    finally:
        cameras.close()
    return 0


def _print_collection_summary(
    *,
    experiment: str,
    task: str,
    state_mode,
    experiment_dir: Path,
    front_crop,
    episode_index: int,
) -> None:
    print()
    info(f"Experiment: {experiment}")
    info(f"Task: {task}")
    info(
        f"Contract: state={state_mode.value}, action={ActionMode.JOINT_ABSOLUTE.value}"
    )
    info(f"Output: {experiment_dir}")
    info(f"AgentView ROI: {'full frame' if front_crop is None else front_crop.xywh}")
    info(f"Next episode: {episode_index}; Ctrl+C quits.")
    print()


def reset_for_next_episode(teleop_config: Path) -> None:
    runtime_script = ROOT_DIR / "scripts/apps/teleop/a1_teleop_runtime.sh"
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


def run_safely(config: TeleopConfig, *, experiment: str) -> int:
    try:
        return run(config, experiment=experiment)
    except (RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 1
