"""Lifecycle for recording one teleop episode and queueing its export."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_ops import (
    EpisodeCaptureReport,
    announce_episode_capture,
    announce_episode_outcome,
)

from galaxea_a1_runtime.apps.teleop.bag_retention import (
    announce_removals,
    enforce_raw_retention,
    remove_raw_episode,
)
from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.export_queue import (
    ExportQueue,
    ExportStats,
    QueuedExport,
)
from galaxea_a1_runtime.apps.teleop.interaction import reset_required_after_recording
from galaxea_a1_runtime.apps.teleop.recording import (
    PreparationMotionError,
    RecordedEpisode,
    record_episode,
)
from galaxea_a1_runtime.collection import EpisodeDecision
from galaxea_a1_runtime.console import failure, success, warning
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
        self._exports = ExportQueue(config)

    def record(
        self,
        episode_index: int,
        *,
        on_recording_ready: Callable[[], None],
        reset_after_save: bool,
    ) -> EpisodeCompletion:
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
            if decision == EpisodeDecision.SAVE:
                self._exports.submit(
                    QueuedExport(
                        episode_index=episode_index,
                        bag_root=recording.bag_root,
                        elapsed_s=recording.elapsed_s,
                    )
                )
            else:
                remove_raw_episode(
                    self.config,
                    recording.bag_root,
                    reason="discarded episode",
                )
                announce_episode_outcome(
                    episode_index=episode_index,
                    decision=decision,
                    frame_count=0,
                    dataset_root=None,
                )
            return EpisodeCompletion(
                decision,
                frame_count=0,
                reset_required=self._reset_required(
                    recording, decision, reset_after_save=reset_after_save
                ),
            )
        except PreparationMotionError as error:
            warning(str(error))
            return EpisodeCompletion(
                EpisodeDecision.DISCARD,
                reset_required=self.config.collection.reset_policy.required_after(
                    EpisodeDecision.DISCARD
                ),
            )
        except BaseException:
            failure(
                "Collection stopped; raw bag retained under data/recordings and previous committed dataset preserved"
            )
            raise

    def poll_exports(self) -> None:
        """Announce finished background exports on the main thread."""

        for outcome in self._exports.take_outcomes():
            decision = (
                EpisodeDecision.SAVE
                if outcome.stored_frames
                else EpisodeDecision.DISCARD
            )
            effective_fps = (
                outcome.sampled_frames / outcome.elapsed_s if outcome.elapsed_s else 0.0
            )
            announce_episode_capture(
                EpisodeCaptureReport(
                    episode_index=outcome.episode_index,
                    sampled_frames=outcome.sampled_frames,
                    stored_frames=outcome.stored_frames,
                    trimmed_frames=outcome.trimmed_frames,
                    elapsed_s=outcome.elapsed_s,
                    effective_fps=effective_fps,
                    decision=decision,
                )
            )
            announce_episode_outcome(
                episode_index=outcome.episode_index,
                decision=decision,
                frame_count=outcome.stored_frames,
                dataset_root=outcome.dataset_root,
            )

    def apply_retention(self) -> None:
        announce_removals(
            enforce_raw_retention(self.config, protect=self._exports.protected_paths())
        )

    def export_failure(self) -> BaseException | None:
        return self._exports.failure()

    def export_stats(self) -> ExportStats:
        return self._exports.stats()

    def drain_exports(self) -> int:
        return self._exports.drain()

    def close_exports(self) -> None:
        self._exports.close()

    @staticmethod
    def _announce_recording_ready(
        episode_index: int,
        callback: Callable[[], None],
    ) -> None:
        success(f"Recording started · episode {episode_index} · move now")
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
