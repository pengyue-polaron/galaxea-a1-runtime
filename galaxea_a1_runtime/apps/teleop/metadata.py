"""Reproducible raw-episode metadata for teleoperation collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from galaxea_a1_runtime.collection import (
    CameraMetadata,
    StateMode,
    TeleopRawEpisodeMetadata,
    metadata_to_json_dict,
    state_names_for_mode,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
from galaxea_a1_runtime.schema import ActionMode, JOINT_ACTION_NAMES


def write_metadata(
    *,
    episode_dir: Path,
    task: str,
    experiment: str,
    episode_index: int,
    frame_count: int,
    fps: float,
    state_mode: StateMode,
    cam0_serial: str | None,
    cam0_width: int,
    cam0_height: int,
    cam0_depth_enabled: bool,
    cam0_depth_width: int,
    cam0_depth_height: int,
    cam0_depth_aligned: bool,
    cam0_source_width: int,
    cam0_source_height: int,
    cam0_crop: ImageRoi | None,
    cam1_label: str,
    cam1_width: int,
    cam1_height: int,
    config_path: str,
    args: argparse.Namespace,
) -> None:
    cameras = [
        CameraMetadata(
            "front",
            "cam0",
            cam0_width,
            cam0_height,
            cam0_serial,
            source_width=cam0_source_width,
            source_height=cam0_source_height,
            crop_xywh=None if cam0_crop is None else cam0_crop.xywh,
        ),
        CameraMetadata("wrist", "cam1", cam1_width, cam1_height, cam1_label),
    ]
    if cam0_depth_enabled:
        cameras.append(
            CameraMetadata(
                "front_depth",
                "cam0_depth",
                cam0_depth_width,
                cam0_depth_height,
                cam0_serial,
                modality="depth",
                dtype="uint16",
                encoding="z16_mm_aligned_to_color" if cam0_depth_aligned else "z16_mm",
                source_width=cam0_source_width,
                source_height=cam0_source_height,
                crop_xywh=None if cam0_crop is None else cam0_crop.xywh,
            )
        )
    metadata = TeleopRawEpisodeMetadata(
        schema_version=TELEOP_RAW_SCHEMA_VERSION,
        collection_mode="teleop",
        task=task,
        experiment=experiment,
        episode_index=episode_index,
        frame_count=frame_count,
        fps_target=fps,
        state_mode=state_mode,
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=state_names_for_mode(state_mode),
        action_names=JOINT_ACTION_NAMES,
        state_topics={
            "joint": args.joint_topic,
            "eef": args.eef_topic,
            "gripper_feedback": args.gripper_feedback_topic,
        },
        action_topics={
            "joint_target": args.action_topic,
            "gripper_target": args.gripper_action_topic,
        },
        control_path=(
            args.action_topic,
            "jointTracker_demo_node",
            args.staged_command_topic,
            "safe_arm_command_relay.py",
            args.host_command_topic,
        ),
        cameras=tuple(cameras),
        config_path=config_path,
        quality_checks={
            "max_joint_action_step_rad": args.max_joint_action_step_rad,
            "max_camera_age_s": args.max_camera_age_s,
            "max_joint_feedback_age_s": args.max_joint_feedback_age_s,
            "max_eef_feedback_age_s": args.max_eef_feedback_age_s,
            "max_action_age_s": args.max_action_age_s,
            "max_gripper_age_s": args.max_gripper_age_s,
            "gripper_continuous_stroke_min_mm": args.gripper_stroke_min,
            "gripper_continuous_stroke_max_mm": args.gripper_stroke_max,
        },
    )
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata_to_json_dict(metadata), indent=2)
    )
