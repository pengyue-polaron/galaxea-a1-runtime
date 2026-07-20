"""Exact OpenPI pi0.5 service/bridge deployment handshake."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.inference.protocol import (
    add_contract_digest,
    validate_exact_metadata,
)
from galaxea_a1_runtime.schema import A1_STATE_NAMES, EEF_ACTION_NAMES


PROTOCOL_VERSION = "galaxea_a1_openpi_pi05_eef_v3"


def server_metadata(config: Pi05Config) -> dict[str, Any]:
    front = required_front_roi(config.system.cameras)
    wrist = config.system.cameras.wrist
    model = config.model
    contract = config.model_contract
    return add_contract_digest(
        {
            "protocol": PROTOCOL_VERSION,
            "deployment_id": config.deployment_id,
            "backend": config.backend.backend_id,
            "code_repository": config.backend.source.repository,
            "code_revision": config.backend.source.revision,
            "environment": {
                "manager": config.backend.environment.manager,
                "python_version": config.backend.environment.python_version,
                "lock_sha256": config.backend.environment.lock_sha256,
            },
            "model_id": model.model_id,
            "model_repo_id": model.source.repo_id,
            "model_revision": model.source.revision,
            "model_revision_label": model.source.revision_label,
            "checkpoint_step": model.checkpoint_step,
            "model_manifest_sha256": model.manifest.sha256,
            "checkpoint_format": contract.checkpoint_format,
            "parameter_set": contract.parameter_set,
            "train_config": contract.train_config,
            "task_catalog": config.task_catalog.protocol_contract(),
            "camera_keys": [
                config.observations.front_key,
                config.observations.wrist_key,
            ],
            "camera_shapes": [
                [front.height, front.width, 3],
                [wrist.height, wrist.width, 3],
            ],
            "policy_image_shape": [224, 224, 3],
            "state_names": list(A1_STATE_NAMES),
            "action_names": list(EEF_ACTION_NAMES),
            "state_shape": [contract.state_dim],
            "action_shape": [contract.action_horizon, contract.source_action_dim],
            "model_action_dim": contract.model_action_dim,
            "pose_mode": contract.pose_mode,
            "gripper_range": [0.0, 1.0],
            "normalization": "quantiles",
            "engine": {
                "jax_platform": config.engine.jax_platform,
                "xla_memory_fraction": config.engine.xla_memory_fraction,
                "seed": config.engine.seed,
                "sampling_steps": config.engine.sampling_steps,
            },
        }
    )


def validate_server_metadata(actual: object, expected: dict[str, Any]) -> None:
    validate_exact_metadata(actual, expected, label="OpenPI pi0.5")
