"""Reproducible provenance for directly recorded LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass

from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.constants import JOINT_TRACKER_NODE_NAME, SAFE_RELAY_SCRIPT
from galaxea_a1_runtime.schema import (
    ACTION_FEATURE_KEY,
    A1_STATE_NAMES,
    JOINT_ACTION_NAMES_RAD,
    STATE_FEATURE_KEY,
    camera_specs_from_system,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


@dataclass(frozen=True)
class DatasetProvenanceRequest:
    task: str
    experiment: str
    front_crop: ImageRoi | None
    wrist_label: str
    config_path: str
    config: TeleopConfig


def build_dataset_provenance(request: DatasetProvenanceRequest) -> dict:
    """Build the A1-specific metadata stored beside standard LeRobot metadata."""

    config = request.config
    system = config.system
    front = system.cameras.front
    crop = request.front_crop
    specs = {spec.name: spec for spec in camera_specs_from_system(system)}
    crop_xywh = None if crop is None else list(crop.xywh)
    cameras = [
        {
            "name": "front",
            "feature_key": specs["front"].feature_key(),
            "width": specs["front"].width,
            "height": specs["front"].height,
            "source": front.serial,
            "source_width": front.width,
            "source_height": front.height,
            "crop_xywh": crop_xywh,
            "modality": "rgb",
        },
        {
            "name": "wrist",
            "feature_key": specs["wrist"].feature_key(),
            "width": specs["wrist"].width,
            "height": specs["wrist"].height,
            "source": request.wrist_label,
            "modality": "rgb",
        },
    ]
    if front.depth:
        depth = specs["front_depth"]
        cameras.append(
            {
                "name": "front_depth",
                "feature_key": depth.feature_key(),
                "width": depth.width,
                "height": depth.height,
                "source": front.serial,
                "source_width": front.width,
                "source_height": front.height,
                "crop_xywh": crop_xywh,
                "modality": "depth",
                "dtype": "uint16",
                "depth_unit": "millimeter",
                "alignment": ("color" if front.align_depth_to_color else "unaligned"),
            }
        )

    return {
        "collection_mode": "teleop",
        "dataset_format": "LeRobotDataset v3.0",
        "experiment": request.experiment,
        "task": request.task,
        "config_path": request.config_path,
        "robot_type": "galaxea_a1",
        "image_storage": "video",
        "observation": {
            "feature": STATE_FEATURE_KEY,
            "names": list(A1_STATE_NAMES),
            "semantics": "absolute EEF pose, measured joints in radians, normalized gripper",
            "eef_reference_frame": "base_link",
            "eef_position_unit": "meter",
            "eef_quaternion_order": "xyzw",
            "joint_unit": "radian",
            "gripper_range": [0.0, 1.0],
        },
        ACTION_FEATURE_KEY: {
            "feature": ACTION_FEATURE_KEY,
            "names": list(JOINT_ACTION_NAMES_RAD),
            "semantics": "absolute joint targets in radians plus normalized gripper target",
            "joint_unit": "radian",
            "gripper_range": [0.0, 1.0],
        },
        "state_topics": {
            "joint": system.topics.joint_states,
            "eef": system.topics.eef_pose,
            "gripper_feedback": system.topics.gripper_feedback,
        },
        "action_topics": {
            "joint_target": system.topics.joint_target,
            "gripper_target": system.topics.gripper_target,
        },
        "control_path": [
            system.topics.joint_target,
            JOINT_TRACKER_NODE_NAME,
            system.topics.staged_command,
            SAFE_RELAY_SCRIPT,
            system.topics.host_command,
        ],
        "cameras": cameras,
        "image_color_space": "RGB",
        "quality_checks": {
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
    }
