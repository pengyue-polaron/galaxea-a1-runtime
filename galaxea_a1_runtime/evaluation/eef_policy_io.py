"""Shared model packets and response validation for offline EEF evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from galaxea_a1_runtime.apps.eef_bridge import condition_state_from_action8
from galaxea_a1_runtime.apps.eef_policy_actions import absolute_action_to_relative
from galaxea_a1_runtime.evaluation.types import EpisodeRecord


def camera_packet(deployment: Any, images: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        deployment.observations.front_key: images["observation.images.front"],
        deployment.observations.wrist_key: images["observation.images.wrist"],
    }


def lingbot_observation(
    deployment: Any,
    episode: EpisodeRecord,
    images: dict[str, np.ndarray],
    frame_index: int,
) -> dict[str, Any]:
    packet = {
        "obs": [camera_packet(deployment, images)],
        "prompt": episode.task,
    }
    absolute = np.concatenate(
        [episode.states[frame_index, :7], episode.states[frame_index, -1:]]
    )
    origin = episode.states[0, :7]
    relative = absolute_action_to_relative(
        absolute,
        origin,
        min_quat_norm=deployment.system.eef.min_quat_norm,
    )
    packet["state"] = condition_state_from_action8(
        relative,
        frame_chunk_size=deployment.policy_server.frame_chunk_size,
        action_per_frame=deployment.policy_server.action_per_frame,
    )
    return packet


def pi05_observation(
    deployment: Any,
    episode: EpisodeRecord,
    images: dict[str, np.ndarray],
    frame_index: int,
) -> dict[str, Any]:
    packet = camera_packet(deployment, images)
    packet["observation/state"] = episode.states[frame_index].copy()
    packet["prompt"] = episode.task
    return packet


def validated_lingbot_action(response: dict[str, Any], deployment: Any) -> np.ndarray:
    action = np.asarray(response.get("action"), dtype=np.float32)
    expected = (
        len(deployment.policy_server.action_channel_ids),
        deployment.policy_server.frame_chunk_size,
        deployment.policy_server.action_per_frame,
    )
    if action.shape != expected or not np.isfinite(action).all():
        raise RuntimeError(
            f"invalid LingBot offline action tensor: expected finite {expected}, "
            f"got {action.shape}"
        )
    return action


def validated_pi05_actions(response: dict[str, Any], deployment: Any) -> np.ndarray:
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    expected = (
        deployment.model_contract.action_horizon,
        deployment.model_contract.source_action_dim,
    )
    if actions.shape != expected or not np.isfinite(actions).all():
        raise RuntimeError(
            f"invalid pi0.5 offline action tensor: expected finite {expected}, "
            f"got {actions.shape}"
        )
    return actions
