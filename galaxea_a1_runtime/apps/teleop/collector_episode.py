"""Lifecycle for recording and committing one teleop episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.metadata import (
    DatasetProvenanceRequest,
    build_dataset_provenance,
)
from galaxea_a1_runtime.apps.teleop.recording import record_episode
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    find_joint_action_step_violation,
    reset_required_after_episode,
)
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.console import failure, info, success, warning
from galaxea_a1_runtime.lerobot.direct_recording import DirectLeRobotEpisode
from galaxea_a1_runtime.schema import (
    JOINT_ACTION_NAMES_RAD,
    camera_specs_from_system,
    canonical_dataset_contract,
)
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
        dataset_root: Path,
        repo_id: str,
        task: str,
        front_crop: ImageRoi | None,
        ros_state: Any,
        cameras: TeleopCameraSession,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.experiment = experiment
        self.dataset_root = dataset_root
        self.repo_id = repo_id
        self.task = task
        self.front_crop = front_crop
        self.ros_state = ros_state
        self.cameras = cameras
        self.repo_root = repo_root
        self.contract = canonical_dataset_contract(
            cameras=camera_specs_from_system(config.system)
        )

    def record(self, episode_index: int) -> EpisodeCompletion:
        warning(
            f"Episode {episode_index} recording: "
            "Enter=save, d+Enter=discard, q+Enter=quit"
        )
        front_reader, wrist_reader = self.cameras.readers
        try:
            with DirectLeRobotEpisode(
                target_root=self.dataset_root,
                repo_id=self.repo_id,
                fps=int(self.config.collection.fps),
                contract=self.contract,
                use_videos=self.config.collection.use_videos,
                experiment=self.experiment,
                task=self.task,
                provenance=self._provenance(),
            ) as output:
                recording = record_episode(
                    dataset=output,
                    task=self.task,
                    front_reader=front_reader,
                    wrist_reader=wrist_reader,
                    ros_state=self.ros_state,
                    fps=self.config.collection.fps,
                    max_duration_s=self.config.collection.max_duration_s,
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
                    return EpisodeCompletion(
                        recording.decision,
                        reset_required=reset_required_after_episode(
                            recording.decision,
                            after_save=self.config.collection.auto_reset_after_save,
                            after_discard=(
                                self.config.collection.auto_reset_after_discard
                            ),
                        ),
                    )

                violation = find_joint_action_step_violation(
                    recording.actions,
                    action_names=JOINT_ACTION_NAMES_RAD,
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
                        reset_required=reset_required_after_episode(
                            EpisodeDecision.DISCARD,
                            after_save=self.config.collection.auto_reset_after_save,
                            after_discard=(
                                self.config.collection.auto_reset_after_discard
                            ),
                        ),
                    )

                output.commit()
        except BaseException:
            failure(
                f"Episode {episode_index}: recording or commit failed; "
                "the previous complete dataset remains authoritative"
            )
            raise
        fps = self.config.collection.fps
        nominal_s = recording.frame_count / fps
        success(
            f"Episode {episode_index} saved: {recording.frame_count} frames "
            f"(~{nominal_s:.1f}s @ {fps:g}fps) -> {self.dataset_root}"
        )
        print()
        return EpisodeCompletion(
            EpisodeDecision.SAVE,
            reset_required=reset_required_after_episode(
                EpisodeDecision.SAVE,
                after_save=self.config.collection.auto_reset_after_save,
                after_discard=self.config.collection.auto_reset_after_discard,
            ),
        )

    def _provenance(self) -> dict:
        return build_dataset_provenance(
            DatasetProvenanceRequest(
                task=self.task,
                experiment=self.experiment,
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
