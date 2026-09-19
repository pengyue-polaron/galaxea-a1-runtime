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

# The tracked Python 3.12 ROS overlay and A1 SDK provide every collector message
# dependency. Do not expose Ubuntu's Python 3.10 site-packages: optional imports
# in Hugging Face datasets can otherwise load ABI-incompatible binary wheels.
configure_ros1_python(ROOT_DIR, include_system_site=False)

import rospy

from embodied_ops import (
    EpisodeDecision,
    announce_collection_session,
    announce_collection_summary,
)
from embodied_ops.operator_panel import announce_input, announce_progress

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.collector_episode import TeleopEpisodeSession
from galaxea_a1_runtime.apps.teleop.dataset_contract import (
    direct_dataset_identity,
)
from galaxea_a1_runtime.apps.teleop.collection_task import (
    normalize_collection_task,
)
from galaxea_a1_runtime.apps.teleop.interaction import (
    A1_COLLECTION_INTERACTION,
    CollectionReadyAction,
    collection_ready_action_ids,
    normalize_collection_ready_action,
)
from galaxea_a1_runtime.apps.teleop.bag_export import (
    export_provenance,
    remove_stale_export_scratch,
)
from galaxea_a1_runtime.apps.teleop.bag_retention import (
    announce_removals,
    enforce_raw_retention,
    raw_recordings_usage,
)
from galaxea_a1_runtime.apps.teleop.export_queue import ExportStats
from galaxea_a1_runtime.apps.teleop.ros_state import RosTeleopState
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.collection import (
    validate_experiment_name,
)
from galaxea_a1_runtime.console import (
    Tone,
    failure,
    info,
    step,
    style,
    success,
    warning,
)
from galaxea_a1_runtime.lerobot.direct_recording import (
    validate_direct_dataset_provenance,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def load_or_prompt_task(
    existing_tasks: tuple[str, ...], *, provided_task: str | None = None
) -> str:
    if provided_task is not None:
        return normalize_collection_task(provided_task)
    if existing_tasks:
        info("Existing experiment prompts:")
        for existing in existing_tasks:
            info(f"  - {existing}")
        info("Enter the exact prompt for the next episode.")
    else:
        info("First run: enter the task prompt.")
    task = input(style("Task > ", Tone.STEP)).strip()
    return normalize_collection_task(task)


def run(config: TeleopConfig, *, experiment: str, task: str | None = None) -> int:
    experiment = validate_experiment_name(experiment)
    front_crop = required_front_roi(config.system.cameras)
    identity = direct_dataset_identity(config, experiment)
    existing = validate_direct_dataset_provenance(
        identity,
        export_provenance(config, experiment),
    )
    task = load_or_prompt_task(existing.tasks, provided_task=task)
    episode_index = existing.total_episodes
    announce_collection_session(
        experiment=experiment,
        task=task,
        repo_id=identity.repo_id,
        dataset_root=str(identity.target_root),
        next_episode=episode_index,
        configuration=(
            "contract=canonical A1 state and absolute joint action "
            f"· agent_view_roi={'full-frame' if front_crop is None else front_crop.xywh}"
        ),
    )

    rospy.init_node("a1_teleop_collect", anonymous=False, disable_signals=True)
    ros_state = RosTeleopState(config)
    step("Waiting for ROS state")
    ros_state.wait_ready(timeout_s=config.collection.ready_timeout_s)
    success("ROS state ready.")
    announce_removals(enforce_raw_retention(config))
    usage = raw_recordings_usage(config)
    if usage is not None:
        used, cap = usage
        info(f"Raw recordings: {used / 1e9:.1f} GB / cap {cap / 1e9:.1f} GB")
    for stale in remove_stale_export_scratch():
        info(f"Removed stale export scratch: {stale}")

    discarded = 0
    interrupted = False
    cameras = TeleopCameraSession(config)
    try:
        step("Starting cameras")
        camera_summary = cameras.start()
        success(f"Cameras ready: {camera_summary}")
        if config.collection.reset_policy.before_collection:
            step("Resetting A1 and leader before the first episode")
            reset_for_next_episode(config.path)
            ros_state.wait_ready(timeout_s=config.collection.ready_timeout_s)
            success("Pre-collection Reset complete; fresh ROS state ready.")
        episodes = TeleopEpisodeSession(
            config=config,
            identity=identity,
            task=task,
            ros_state=ros_state,
            cameras=cameras,
        )
        reset_after_save = config.collection.reset_policy.after_save
        while not rospy.is_shutdown():
            episodes.poll_exports()
            episodes.apply_retention()
            _raise_export_failure(episodes)
            status_detail = _collection_status_detail(
                episode_index=episode_index,
                task=task,
                stats=episodes.export_stats(),
            )
            announce_progress(
                "collection",
                "Collection episode",
                episode_index,
                None,
                phase="ready",
                detail=status_detail,
                force=True,
            )
            ready_action_ids = collection_ready_action_ids(
                reset_after_save=reset_after_save
            )
            announce_input(
                ready_action_ids,
                phase="ready",
                detail=(
                    f"{status_detail} · Reset after save="
                    f"{'on' if reset_after_save else 'off'}"
                ),
            )
            command = (
                input(
                    style(
                        A1_COLLECTION_INTERACTION.start_prompt(episode_index),
                        Tone.STEP,
                    )
                )
                .strip()
                .lower()
            )
            ready_action = normalize_collection_ready_action(command)
            if ready_action is CollectionReadyAction.QUIT:
                break
            if ready_action is CollectionReadyAction.ENABLE_RESET_AFTER_SAVE:
                reset_after_save = True
                continue
            if ready_action is CollectionReadyAction.DISABLE_RESET_AFTER_SAVE:
                reset_after_save = False
                continue
            if ready_action is CollectionReadyAction.RESET:
                announce_progress(
                    "collection",
                    "Collection episode",
                    episode_index,
                    None,
                    phase="resetting",
                    detail=status_detail,
                    force=True,
                )
                reset_for_next_episode(config.path)
                ros_state.wait_ready(timeout_s=config.collection.ready_timeout_s)
                continue
            episodes.poll_exports()
            _raise_export_failure(episodes)
            status_detail = _collection_status_detail(
                episode_index=episode_index,
                task=task,
                stats=episodes.export_stats(),
            )
            announce_progress(
                "collection",
                "Collection episode",
                episode_index,
                None,
                phase="preparing",
                detail=status_detail,
                force=True,
            )

            def announce_recording_ready() -> None:
                announce_progress(
                    "collection",
                    "Collection episode",
                    episode_index,
                    None,
                    phase="recording",
                    detail=status_detail,
                    force=True,
                )
                announce_input(
                    A1_COLLECTION_INTERACTION.recording_action_ids,
                    phase="recording",
                    detail=status_detail,
                )

            completion = episodes.record(
                episode_index,
                on_recording_ready=announce_recording_ready,
                reset_after_save=reset_after_save,
            )
            episodes.poll_exports()
            if completion.decision == EpisodeDecision.QUIT:
                break
            if completion.decision == EpisodeDecision.SAVE:
                episode_index += 1
            elif completion.decision == EpisodeDecision.DISCARD:
                discarded += 1
            if completion.reset_required:
                announce_progress(
                    "collection",
                    "Collection episode",
                    episode_index,
                    None,
                    phase="resetting",
                    detail=_collection_status_detail(
                        episode_index=episode_index,
                        task=task,
                        stats=episodes.export_stats(),
                    ),
                    force=True,
                )
                reset_for_next_episode(config.path)
    except (KeyboardInterrupt, EOFError):
        print()
        interrupted = True
    finally:
        cameras.close()
    if not interrupted and not rospy.is_shutdown():
        _raise_export_failure(episodes)
        pending = episodes.drain_exports()
        episodes.poll_exports()
        stats = episodes.export_stats()
        if pending:
            warning(
                f"{pending} dataset exports did not finish; "
                "raw bags are retained for a later bag-export"
            )
        _raise_export_failure(episodes)
    else:
        stats = episodes.export_stats()
        if stats.pending:
            warning(
                f"{stats.pending} dataset exports are still running; "
                "raw bags are retained and can be exported later"
            )
    episodes.close_exports()
    announce_collection_summary(
        saved=stats.exported,
        discarded=discarded + stats.rejected,
        saved_frames=stats.stored_frames,
    )
    announce_progress(
        "collection",
        "Collection episode",
        episode_index,
        None,
        phase="completed",
        detail=(
            f"exported={stats.exported} · "
            f"discarded={discarded + stats.rejected} · frames={stats.stored_frames}"
        ),
        force=True,
    )
    return 0


def _raise_export_failure(episodes: TeleopEpisodeSession) -> None:
    exc = episodes.export_failure()
    if exc is None:
        return
    raise RuntimeError(
        "background dataset export failed; collection stopped for inspection "
        f"(raw bag retained): {exc}"
    )


def _collection_status_detail(
    *, episode_index: int, task: str, stats: ExportStats
) -> str:
    exporting = f" · {stats.pending} exporting" if stats.pending else ""
    return (
        f"Episode {episode_index} · {stats.exported} exported{exporting} · "
        f"{stats.stored_frames} frames · {task}"
    )


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
