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
from galaxea_a1_runtime.apps.teleop.recording import RecordedEpisode, record_episode
from galaxea_a1_runtime.collection import EpisodeDecision
from galaxea_a1_runtime.console import failure, warning
from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectDatasetIdentity,
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
        ros_state: Any,
        cameras: TeleopCameraSession,
    ) -> None:
        self.config = config
        self.identity = identity
        self.task = task
        self.ros_state = ros_state
        self.cameras = cameras

    def record(
        self,
        episode_index: int,
        *,
        on_recording_ready: Callable[[], None],
        reset_after_save: bool,
    ) -> EpisodeCompletion:
        from galaxea_a1_runtime.apps.teleop.bag_export import export_bag

        try:
            recording = record_episode(
                config=self.config,
                experiment=self.identity.experiment,
                task=self.task,
                cameras=self.cameras,
                ros_state=self.ros_state,
                on_ready=lambda: self._announce_recording_ready(
                    episode_index, on_recording_ready
                ),
            )
            decision = recording.decision
            frames = 0
            if decision == EpisodeDecision.SAVE:
                announce_progress(
                    "collection",
                    "Collection episode",
                    episode_index,
                    None,
                    phase="saving",
                    detail="Raw bag finalized; aligning and exporting dataset",
                    force=True,
                )
                result = export_bag(recording.bag_root, config=self.config)
                frames = result.stored_frames
                if not frames:
                    decision = EpisodeDecision.DISCARD
                announce_episode_capture(
                    EpisodeCaptureReport(
                        episode_index=episode_index,
                        sampled_frames=result.sampled_frames,
                        stored_frames=frames,
                        trimmed_frames=result.trimmed_frames,
                        elapsed_s=recording.elapsed_s,
                        effective_fps=result.sampled_frames / recording.elapsed_s,
                        decision=decision,
                    )
                )
            announce_episode_outcome(
                episode_index=episode_index,
                decision=decision,
                frame_count=frames,
                dataset_root=str(self.identity.target_root) if frames else None,
            )
            return EpisodeCompletion(
                decision,
                frame_count=frames,
                reset_required=self._reset_required(
                    recording, decision, reset_after_save=reset_after_save
                ),
            )
        except PublishedOutputCleanupError as error:
            warning(
                f"Dataset saved at {error.target}; displaced backup retained at {error.backup}"
            )
            raise
        except BaseException:
            failure(
                "Collection stopped; raw bag retained under data/recordings and previous committed dataset preserved"
            )
            raise

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
