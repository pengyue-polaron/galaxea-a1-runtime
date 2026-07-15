"""Reproducible raw-episode metadata for teleoperation collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.collection import (
    CameraMetadata,
    StateMode,
    TeleopRawEpisodeMetadata,
    metadata_to_json_dict,
    state_names_for_mode,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.constants import JOINT_TRACKER_NODE_NAME, SAFE_RELAY_SCRIPT
from galaxea_a1_runtime.filesystem import atomic_write_text
from galaxea_a1_runtime.schema import (
    ActionMode,
    JOINT_ACTION_NAMES,
    camera_specs_from_system,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


@dataclass(frozen=True)
class EpisodeMetadataRequest:
    episode_dir: Path
    task: str
    experiment: str
    episode_index: int
    frame_count: int
    state_mode: StateMode
    front_crop: ImageRoi | None
    wrist_label: str
    config_path: str
    config: TeleopConfig


def write_metadata(request: EpisodeMetadataRequest) -> None:
    config = request.config
    system = config.system
    front = system.cameras.front
    crop = request.front_crop
    specs = {spec.name: spec for spec in camera_specs_from_system(system)}
    front_spec = specs["front"]
    wrist_spec = specs["wrist"]
    crop_xywh = None if crop is None else crop.xywh

    cameras = [
        CameraMetadata(
            "front",
            "cam0",
            front_spec.width,
            front_spec.height,
            front.serial,
            source_width=front.width,
            source_height=front.height,
            crop_xywh=crop_xywh,
        ),
        CameraMetadata(
            "wrist",
            "cam1",
            wrist_spec.width,
            wrist_spec.height,
            request.wrist_label,
        ),
    ]
    if front.depth:
        depth_spec = specs["front_depth"]
        cameras.append(
            CameraMetadata(
                "front_depth",
                "cam0_depth",
                depth_spec.width,
                depth_spec.height,
                front.serial,
                modality="depth",
                dtype="uint16",
                encoding=(
                    "z16_mm_aligned_to_color"
                    if front.align_depth_to_color
                    else "z16_mm"
                ),
                source_width=front.width,
                source_height=front.height,
                crop_xywh=crop_xywh,
            )
        )

    metadata = TeleopRawEpisodeMetadata(
        schema_version=TELEOP_RAW_SCHEMA_VERSION,
        collection_mode="teleop",
        task=request.task,
        experiment=request.experiment,
        episode_index=request.episode_index,
        frame_count=request.frame_count,
        fps_target=config.collection.fps,
        state_mode=request.state_mode,
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=state_names_for_mode(request.state_mode),
        action_names=JOINT_ACTION_NAMES,
        state_topics={
            "joint": system.topics.joint_states,
            "eef": system.topics.eef_pose,
            "gripper_feedback": system.topics.gripper_feedback,
        },
        action_topics={
            "joint_target": system.topics.joint_target,
            "gripper_target": system.topics.gripper_target,
        },
        control_path=(
            system.topics.joint_target,
            JOINT_TRACKER_NODE_NAME,
            system.topics.staged_command,
            SAFE_RELAY_SCRIPT,
            system.topics.host_command,
        ),
        cameras=tuple(cameras),
        config_path=request.config_path,
        quality_checks={
            "max_joint_action_step_rad": config.collection.max_joint_action_step_rad,
            "max_camera_age_s": system.cameras.max_age_s,
            "max_camera_pair_skew_s": system.cameras.max_pair_skew_s,
            "max_joint_feedback_age_s": system.joint_safety.max_feedback_age_s,
            "max_eef_feedback_age_s": system.eef.max_feedback_age_s,
            "max_action_age_s": system.joint_safety.max_feedback_age_s,
            "max_gripper_age_s": system.joint_safety.max_feedback_age_s,
            "leader_gripper_source_min": config.gripper.source_min,
            "leader_gripper_source_max": config.gripper.source_max,
            "gripper_continuous_stroke_min_mm": system.gripper.stroke_min_mm,
            "gripper_continuous_stroke_max_mm": system.gripper.stroke_max_mm,
        },
    )
    atomic_write_text(
        request.episode_dir / "metadata.json",
        json.dumps(metadata_to_json_dict(metadata), indent=2) + "\n",
    )
