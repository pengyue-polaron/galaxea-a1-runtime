"""Lifecycle for recording and committing one teleop episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.metadata import (
    EpisodeMetadataRequest,
    write_metadata,
)
from galaxea_a1_runtime.apps.teleop.recording import record_episode
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    StateMode,
    find_joint_action_step_violation,
)
from galaxea_a1_runtime.collection.episode_output import validate_staged_episode
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.console import failure, info, success, warning
from galaxea_a1_runtime.filesystem import OutputDirectoryTransaction
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


@dataclass(frozen=True)
class EpisodeCompletion:
    decision: EpisodeDecision
    reset_required: bool = False


class TeleopEpisodeSession:
    def __init__(
        self,
        *,
        config: TeleopConfig,
        experiment: str,
        experiment_dir: Path,
        task: str,
        state_mode: StateMode,
        front_crop: ImageRoi | None,
        ros_state: Any,
        cameras: TeleopCameraSession,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.experiment = experiment
        self.experiment_dir = experiment_dir
        self.task = task
        self.state_mode = state_mode
        self.front_crop = front_crop
        self.ros_state = ros_state
        self.cameras = cameras
        self.repo_root = repo_root

    def record(self, episode_index: int) -> EpisodeCompletion:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{episode_index:03d}_{timestamp}"
        episode_dir = self.experiment_dir / episode_name
        warning(
            f"Episode {episode_index} recording: "
            "Enter=save, d+Enter=discard, q+Enter=quit"
        )
        front_reader, wrist_reader = self.cameras.readers
        try:
            with OutputDirectoryTransaction(episode_dir) as transaction:
                staging_dir = transaction.path
                if staging_dir is None:
                    raise RuntimeError("episode output transaction did not start")
                recording = record_episode(
                    episode_dir=staging_dir,
                    front_reader=front_reader,
                    wrist_reader=wrist_reader,
                    ros_state=self.ros_state,
                    state_mode=self.state_mode,
                    fps=self.config.collection.fps,
                    max_duration_s=self.config.collection.max_duration_s,
                    jpeg_quality=self.config.collection.jpeg_quality,
                    depth_enabled=self.config.system.cameras.front.depth,
                    front_crop=self.front_crop,
                    camera_ready_timeout_s=self.config.collection.ready_timeout_s,
                    max_camera_age_s=self.config.system.cameras.max_age_s,
                    max_camera_pair_skew_s=(self.config.system.cameras.max_pair_skew_s),
                )

                if (
                    recording.decision != EpisodeDecision.SAVE
                    or recording.frame_count == 0
                ):
                    reason = (
                        "0 frames"
                        if recording.frame_count == 0
                        else f"user selected {recording.decision.value}"
                    )
                    info(f"Episode {episode_index}: {reason}; staging output removed.")
                    print()
                    return EpisodeCompletion(recording.decision)

                violation = find_joint_action_step_violation(
                    recording.actions,
                    action_names=JOINT_ACTION_NAMES,
                    max_step_rad=self.config.collection.max_joint_action_step_rad,
                )
                if violation is not None:
                    failure(
                        f"Episode {episode_index} rejected: joint action discontinuity: "
                        f"{violation.describe()}"
                    )
                    failure(
                        f"Episode {episode_index} staging output removed; index will be reused."
                    )
                    print()
                    return EpisodeCompletion(
                        EpisodeDecision.DISCARD,
                        reset_required=self.config.collection.auto_reset_after_save,
                    )

                self._write_metadata(
                    episode_dir=staging_dir,
                    episode_index=episode_index,
                    frame_count=recording.frame_count,
                )
                validate_staged_episode(
                    staging_dir,
                    frame_count=recording.frame_count,
                    depth_enabled=self.config.system.cameras.front.depth,
                )
                transaction.commit()
        except BaseException:
            failure(
                f"Episode {episode_index}: recording or commit failed; "
                f"no episode was committed -> {episode_name}"
            )
            raise
        fps = self.config.collection.fps
        nominal_s = recording.frame_count / fps
        success(
            f"Episode {episode_index} saved: {recording.frame_count} frames "
            f"(~{nominal_s:.1f}s @ {fps:g}fps) -> {episode_name}"
        )
        print()
        return EpisodeCompletion(
            EpisodeDecision.SAVE,
            reset_required=self.config.collection.auto_reset_after_save,
        )

    def _write_metadata(
        self,
        *,
        episode_dir: Path,
        episode_index: int,
        frame_count: int,
    ) -> None:
        write_metadata(
            EpisodeMetadataRequest(
                episode_dir=episode_dir,
                task=self.task,
                experiment=self.experiment,
                episode_index=episode_index,
                frame_count=frame_count,
                state_mode=self.state_mode,
                front_crop=self.front_crop,
                wrist_label=self.cameras.wrist_label,
                config_path=self._config_reference(),
                config=self.config,
            )
        )

    def _config_reference(self) -> str:
        resolved = self.config.path.resolve()
        try:
            return resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(resolved)
