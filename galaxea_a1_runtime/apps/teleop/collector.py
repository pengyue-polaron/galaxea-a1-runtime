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

from operator_panel.protocol import announce_input

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.collector_episode import TeleopEpisodeSession
from galaxea_a1_runtime.apps.teleop.dataset_contract import (
    direct_dataset_identity,
    tracked_config_reference,
)
from galaxea_a1_runtime.apps.teleop.collection_task import (
    prepare_collection_task,
    read_collection_task,
)
from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    normalize_episode_decision,
    validate_experiment_name,
)
from galaxea_a1_runtime.console import Tone, failure, info, step, style, success
from galaxea_a1_runtime.lerobot.direct_recording import inspect_direct_dataset
from galaxea_a1_runtime.schema import ActionMode
from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def load_or_prompt_task(
    experiment_dir: Path, *, provided_task: str | None = None
) -> str:
    if provided_task is not None:
        return prepare_collection_task(experiment_dir, provided_task)
    if existing := read_collection_task(experiment_dir):
        return existing
    info("First run: enter the task prompt.")
    task = input(style("Task > ", Tone.STEP)).strip()
    return prepare_collection_task(experiment_dir, task)


def run(config: TeleopConfig, *, experiment: str, task: str | None = None) -> int:
    experiment = validate_experiment_name(experiment)
    state_mode = StateMode.EEF_JOINT
    front_crop = required_front_roi(config.system.cameras)
    identity = direct_dataset_identity(config, experiment)
    config_reference = tracked_config_reference(config, repo_root=ROOT_DIR)
    existing = inspect_direct_dataset(
        identity,
    )
    task = load_or_prompt_task(identity.target_root, provided_task=task)
    if existing.task is not None and existing.task != task:
        raise ValueError(
            f"collection task mismatch for {experiment}: "
            f"existing={existing.task!r}, requested={task!r}"
        )

    rospy.init_node("a1_teleop_collect", anonymous=False, disable_signals=True)
    ros_state = RosTeleopState(config)
    step("Waiting for ROS state")
    ros_state.wait_ready(
        state_mode=state_mode, timeout_s=config.collection.ready_timeout_s
    )
    success("ROS state ready.")

    episode_index = existing.total_episodes
    cameras = TeleopCameraSession(config)
    try:
        step("Starting cameras")
        camera_summary = cameras.start()
        success(f"Cameras ready: {camera_summary}")
        _print_collection_summary(
            experiment=experiment,
            task=task,
            state_mode=state_mode,
            dataset_root=identity.target_root,
            repo_id=identity.repo_id,
            front_crop=front_crop,
            episode_index=episode_index,
        )
        episodes = TeleopEpisodeSession(
            config=config,
            identity=identity,
            task=task,
            front_crop=front_crop,
            ros_state=ros_state,
            cameras=cameras,
            config_reference=config_reference,
        )
        while not rospy.is_shutdown():
            announce_input(("enter", "quit"))
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
            announce_input(("enter", "discard", "quit"))
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
    dataset_root: Path,
    repo_id: str,
    front_crop,
    episode_index: int,
) -> None:
    print()
    info(f"Experiment: {experiment}")
    info(f"Task: {task}")
    info(
        f"Contract: state={state_mode.value}, action={ActionMode.JOINT_ABSOLUTE.value}"
    )
    info(f"LeRobot repo ID: {repo_id}")
    info(f"Output: {dataset_root}")
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


def run_safely(
    config: TeleopConfig, *, experiment: str, task: str | None = None
) -> int:
    try:
        return run(config, experiment=experiment, task=task)
    except (RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 1
