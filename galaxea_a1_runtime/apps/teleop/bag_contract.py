"""Shared identity and topic contract for authoritative collection bags."""

from __future__ import annotations

import hashlib
from pathlib import Path

from galaxea_a1_runtime.apps.cameras.ros2_pipeline import image_topic, CAMERA_NAMESPACE

from galaxea_a1_runtime.schema import BAG_CAPTURE_CONTRACT

CAPTURE_CONTRACT = BAG_CAPTURE_CONTRACT
MANIFEST_NAME = "episode.json"


def source_topics(system) -> dict[str, str]:
    topics = system.topics
    mirrors = system.observability.topics
    result = {
        "front": image_topic("front"),
        "wrist": image_topic("wrist"),
        "joints": topics.joint_states,
        "eef": topics.eef_pose,
        "gripper": mirrors.gripper_feedback_state,
        "action": topics.joint_target,
        "gripper_action": mirrors.gripper_target_state,
    }
    if system.cameras.front.depth:
        result["depth"] = f"{CAMERA_NAMESPACE}/front/aligned_depth_to_color/image_raw"
    return result


def config_fingerprint(config) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("teleop", config.path), ("system", config.system.path))
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
