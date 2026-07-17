"""Pure LingBot server/bridge deployment contract handshake."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.inference.protocol import (
    add_contract_digest,
    validate_exact_metadata,
)


PROTOCOL_VERSION = "galaxea_a1_lingbot_eef_v1"


def server_metadata(config: LingBotConfig) -> dict[str, Any]:
    policy = config.policy_server
    front_roi = required_front_roi(config.system.cameras)
    wrist = config.system.cameras.wrist
    contract: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "backend": policy.backend.backend_id,
        "code_repository": policy.backend.source.repository,
        "code_revision": policy.backend.source.revision,
        "environment": {
            "manager": policy.backend.environment.manager,
            "python_version": policy.backend.environment.python_version,
            "lock_sha256": policy.backend.environment.lock_sha256,
        },
        "model_repo_id": policy.model.source.repo_id,
        "model_revision": policy.model.source.revision,
        "model_artifact": {
            "manifest_sha256": policy.model.manifest.sha256,
            "transformer_weight_sha256": policy.expected_weight_sha256,
            "transformer_config_sha256": policy.expected_transformer_config_sha256,
        },
        "vendor_config": policy.vendor_config,
        "prompt": config.server.prompt,
        "camera_keys": [
            config.observations.front_key,
            config.observations.wrist_key,
        ],
        "camera_shapes": [
            [front_roi.height, front_roi.width, 3],
            [wrist.height, wrist.width, 3],
        ],
        "policy_image_shape": [policy.height, policy.width, 3],
        "action_shape": [8, policy.frame_chunk_size, policy.action_per_frame],
        "model_action_dim": policy.model_action_dim,
        "action_channel_ids": list(policy.action_channel_ids),
        "pose_mode": config.action.pose_mode,
        "gripper_range": [0.0, 1.0],
        "normalization": {
            "method": "quantiles",
            "q01_source": list(policy.q01_source),
            "q99_source": list(policy.q99_source),
        },
        "attention_mode": policy.attention_mode,
        "enable_offload": policy.enable_offload,
        "text_encoder_device": policy.text_encoder_device,
        "parallelism": {
            "world_size": policy.world_size,
            "fsdp": False,
        },
        "temporal_cache": {
            "observations_per_action_frame": (
                config.execution.kv_observations_per_frame
            ),
            "actions_per_observation": (
                policy.action_per_frame // config.execution.kv_observations_per_frame
            ),
        },
        "inference": {
            "seed": policy.seed,
            "attention_window": policy.attention_window,
            "guidance_scale": policy.guidance_scale,
            "action_guidance_scale": policy.action_guidance_scale,
            "video_inference_steps": policy.video_inference_steps,
            "action_inference_steps": policy.action_inference_steps,
            "snr_shift": policy.snr_shift,
            "action_snr_shift": policy.action_snr_shift,
        },
    }
    return add_contract_digest(contract)


def validate_server_metadata(
    actual: object,
    expected: dict[str, Any],
) -> None:
    validate_exact_metadata(actual, expected, label="LingBot")
