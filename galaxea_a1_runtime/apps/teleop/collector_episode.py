"""Lifecycle for recording and committing one teleop episode."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.teleop.collector_camera import TeleopCameraSession
from galaxea_a1_runtime.apps.teleop.metadata import write_metadata
from galaxea_a1_runtime.apps.teleop.recording import record_episode
from galaxea_a1_runtime.collection import (
    EpisodeDecision,
    StateMode,
    find_joint_action_step_violation,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES


@dataclass(frozen=True)
class EpisodeCompletion:
    decision: EpisodeDecision
    reset_required: bool = False


class TeleopEpisodeSession:
    def __init__(
        self,
        *,
        args: Any,
        experiment_dir: Path,
        task: str,
        state_mode: StateMode,
        front_crop: ImageRoi | None,
        ros_state: Any,
        cameras: TeleopCameraSession,
        repo_root: Path,
    ) -> None:
        self.args = args
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
        print(
            styled(f"  [{episode_index}] RECORDING", "1;33")
            + "  Enter=save, d+Enter=discard, q+Enter=quit"
        )
        front_reader, wrist_reader = self.cameras.readers
        try:
            recording = record_episode(
                episode_dir=episode_dir,
                front_reader=front_reader,
                wrist_reader=wrist_reader,
                ros_state=self.ros_state,
                state_mode=self.state_mode,
                fps=self.args.fps,
                max_duration_s=self.args.max_duration_s,
                jpeg_quality=self.args.jpeg_quality,
                depth_enabled=self.args.cam0_depth_enabled,
                front_crop=self.front_crop,
                camera_ready_timeout_s=self.args.ready_timeout_s,
                max_camera_age_s=self.args.max_camera_age_s,
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
            return EpisodeCompletion(recording.decision)

        violation = find_joint_action_step_violation(
            recording.actions,
            action_names=JOINT_ACTION_NAMES,
            max_step_rad=self.args.max_joint_action_step_rad,
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
            return EpisodeCompletion(
                EpisodeDecision.DISCARD,
                reset_required=self.args.auto_reset_after_save,
            )

        self._write_metadata(
            episode_dir=episode_dir,
            episode_index=episode_index,
            frame_count=recording.frame_count,
        )
        nominal_s = recording.frame_count / self.args.fps
        print(
            styled(
                f"  [{episode_index}] SAVED {recording.frame_count} frames "
                f"(~{nominal_s:.1f}s @ {self.args.fps:g}fps) -> {episode_name}",
                "1;32",
            )
            + "\n"
        )
        return EpisodeCompletion(
            EpisodeDecision.SAVE,
            reset_required=self.args.auto_reset_after_save,
        )

    def _write_metadata(
        self,
        *,
        episode_dir: Path,
        episode_index: int,
        frame_count: int,
    ) -> None:
        crop = self.front_crop
        color_width = self.args.cam0_width if crop is None else crop.width
        color_height = self.args.cam0_height if crop is None else crop.height
        depth_width = (
            crop.width
            if crop is not None
            else (
                self.args.cam0_width
                if self.args.cam0_align_depth_to_color
                else self.args.cam0_depth_width
            )
        )
        depth_height = (
            crop.height
            if crop is not None
            else (
                self.args.cam0_height
                if self.args.cam0_align_depth_to_color
                else self.args.cam0_depth_height
            )
        )
        write_metadata(
            episode_dir=episode_dir,
            task=self.task,
            experiment=self.args.experiment,
            episode_index=episode_index,
            frame_count=frame_count,
            fps=self.args.fps,
            state_mode=self.state_mode,
            cam0_serial=self.args.cam0_serial,
            cam0_width=color_width,
            cam0_height=color_height,
            cam0_depth_enabled=self.args.cam0_depth_enabled,
            cam0_depth_width=depth_width,
            cam0_depth_height=depth_height,
            cam0_depth_aligned=self.args.cam0_align_depth_to_color,
            cam0_source_width=self.args.cam0_width,
            cam0_source_height=self.args.cam0_height,
            cam0_crop=crop,
            cam1_label=self.cameras.wrist_label,
            cam1_width=self.args.cam1_width,
            cam1_height=self.args.cam1_height,
            config_path=self._config_reference(),
            args=self.args,
        )

    def _config_reference(self) -> str:
        resolved = self.args.teleop_config.resolve()
        try:
            return resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(resolved)


def styled(text: str, code: str, *, stream: Any = sys.stdout) -> str:
    if not stream.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"
