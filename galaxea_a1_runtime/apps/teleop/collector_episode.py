"""Lifecycle for recording and committing one teleop episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_ops import (
    EpisodeCaptureReport,
    announce_episode_capture,
    announce_episode_outcome,
)
from embodied_ops.artifacts import PublishedOutputCleanupError
from embodied_ops.operator_panel import announce_progress

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.interaction import (
    collection_recording_notice,
    reset_required_after_recording,
)
from galaxea_a1_runtime.apps.teleop.metadata import (
    DatasetProvenanceRequest,
    build_dataset_provenance,
)
from galaxea_a1_runtime.apps.teleop.recording import RecordedEpisode, record_episode
from galaxea_a1_runtime.collection import EpisodeDecision
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.console import failure, warning
from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectDatasetIdentity,
    DirectLeRobotEpisode,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


@dataclass(frozen=True)
class EpisodeCompletion:
    decision: EpisodeDecision
    frame_count: int = 0
    reset_required: bool = False


class TeleopEpisodeSession:
    def __init__(
        self,
        *,
        config: TeleopConfig,
        identity: DirectDatasetIdentity,
        task: str,
        front_crop: ImageRoi | None,
        ros_state: Any,
        cameras: TeleopCameraSession,
        config_reference: str,
    ) -> None:
        self.config = config
        self.identity = identity
        self.task = task
        self.front_crop = front_crop
        self.ros_state = ros_state
        self.cameras = cameras
        self.config_reference = config_reference

    def record(
        self,
        episode_index: int,
        *,
        on_recording_ready: Callable[[], None],
        reset_after_save: bool,
    ) -> EpisodeCompletion:
        front_reader, wrist_reader = self.cameras.readers
        try:
            with DirectLeRobotEpisode(
                identity=self.identity,
                task=self.task,
                provenance=self._provenance(),
            ) as output:
                recording = record_episode(
                    episode_index=episode_index,
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
                    max_camera_pair_skew_s=self.config.system.cameras.max_pair_skew_s,
                    leading_stillness=self.config.collection.leading_stillness,
                    on_ready=lambda: self._announce_recording_ready(
                        episode_index,
                        on_recording_ready,
                    ),
                )
                report = EpisodeCaptureReport(
                    episode_index=episode_index,
                    sampled_frames=recording.sampled_frame_count,
                    stored_frames=recording.frame_count,
                    trimmed_frames=recording.trimmed_frame_count,
                    elapsed_s=recording.elapsed_s,
                    effective_fps=recording.effective_fps,
                    decision=recording.decision,
                )
                announce_episode_capture(report)

                if (
                    recording.decision != EpisodeDecision.SAVE
                    or recording.frame_count == 0
                ):
                    decision = (
                        EpisodeDecision.DISCARD
                        if recording.frame_count == 0
                        else recording.decision
                    )
                    announce_progress(
                        "collection",
                        "Collection episode",
                        episode_index,
                        None,
                        phase=(
                            "discarding"
                            if decision is EpisodeDecision.DISCARD
                            else "stopping"
                        ),
                        detail=(
                            f"Episode {episode_index} · no motion detected"
                            if recording.frame_count == 0
                            else f"Episode {episode_index} · {recording.frame_count} frames"
                        ),
                        force=True,
                    )
                    announce_episode_outcome(
                        episode_index=episode_index,
                        decision=decision,
                        frame_count=recording.frame_count,
                        dataset_root=None,
                    )
                    print()
                    return EpisodeCompletion(
                        decision,
                        frame_count=recording.frame_count,
                        reset_required=self._reset_required(
                            recording,
                            decision,
                            reset_after_save=reset_after_save,
                        ),
                    )

                announce_progress(
                    "collection",
                    "Collection episode",
                    episode_index,
                    None,
                    phase="saving",
                    detail=f"Episode {episode_index} · {recording.frame_count} frames",
                    force=True,
                )
                output.commit()
        except PublishedOutputCleanupError as error:
            warning(
                f"Episode {episode_index} was saved to {error.target}, but the displaced "
                f"backup could not be removed: {error.backup}"
            )
            failure(
                "Collection stopped: the saved dataset is authoritative; "
                "inspect the backup before retrying."
            )
            raise
        except BaseException:
            failure(
                f"Episode {episode_index}: recording or commit failed; "
                "the previous complete dataset remains authoritative"
            )
            raise
        announce_episode_outcome(
            episode_index=episode_index,
            decision=EpisodeDecision.SAVE,
            frame_count=recording.frame_count,
            dataset_root=str(self.identity.target_root),
        )
        print()
        return EpisodeCompletion(
            EpisodeDecision.SAVE,
            frame_count=recording.frame_count,
            reset_required=self._reset_required(
                recording,
                EpisodeDecision.SAVE,
                reset_after_save=reset_after_save,
            ),
        )

    @staticmethod
    def _announce_recording_ready(
        episode_index: int,
        callback: Callable[[], None],
    ) -> None:
        warning(collection_recording_notice(episode_index))
        callback()

    def _reset_required(
        self,
        recording: RecordedEpisode,
        decision: EpisodeDecision,
        *,
        reset_after_save: bool,
    ) -> bool:
        return reset_required_after_recording(
            decision,
            policy=self.config.collection.reset_policy,
            override=recording.reset_required_override,
            reset_after_save=reset_after_save,
        )

    def _provenance(self) -> dict:
        return build_dataset_provenance(
            DatasetProvenanceRequest(
                experiment=self.identity.experiment,
                front_crop=self.front_crop,
                wrist_label=self.cameras.wrist_label,
                config_path=self.config_reference,
                config=self.config,
            )
        )
