from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.protocol import (
    PROTOCOL_VERSION,
    project_gripper_quantile_latent,
    server_metadata,
    validate_server_metadata,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/deployments/lingbot/fruit_placement_eef.toml"


def test_server_metadata_exhaustively_identifies_deployment_contract():
    config = load_lingbot_config(CONFIG, repo_root=REPO)

    metadata = server_metadata(config)

    assert metadata["protocol"] == PROTOCOL_VERSION
    assert metadata["environment"]["lock_sha256"] == (
        config.policy_server.backend.environment.lock_sha256
    )
    assert metadata["model_revision"] == config.policy_server.model.source.revision
    assert metadata["model_artifact"]["transformer_weight_sha256"] == (
        config.policy_server.expected_weight_sha256
    )
    assert metadata["task_catalog"] == config.task_catalog.protocol_contract()
    assert metadata["camera_keys"] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert metadata["camera_shapes"] == [[480, 480, 3], [480, 640, 3]]
    assert metadata["action_shape"] == [8, 4, 4]
    assert metadata["action_channel_ids"] == [0, 1, 2, 3, 4, 5, 6, 28]
    assert metadata["normalization"]["gripper_latent_projection"] == {
        "quantile_interval": [-1.0, 1.0],
        "reject_limit": 1.5,
    }
    assert metadata["parallelism"] == {"world_size": 1, "fsdp": False}
    assert metadata["attention_capture"] == {
        "request_key": "capture_attention",
        "layers": list(range(30)),
        "map_kind": "wam_multistage_cache_aware_attention_rollout",
    }
    assert metadata["temporal_cache"] == {
        "observations_per_action_frame": 4,
        "actions_per_observation": 1,
    }
    assert metadata["inference"]["action_inference_steps"] == 10
    assert len(metadata["contract_sha256"]) == 64


def test_server_metadata_rejects_any_contract_drift():
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    expected = server_metadata(config)
    actual = dict(expected)
    actual["action_shape"] = [8, 4, 20]

    with pytest.raises(RuntimeError, match="contract mismatch: action_shape"):
        validate_server_metadata(actual, expected)


def test_gripper_latent_projection_handles_observed_release_overshoot():
    q01 = 0.057738296687603
    q99 = 1.0
    observed_release = 1.0147238969802856
    observed_latent = (observed_release - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    projected = project_gripper_quantile_latent(
        [[observed_latent, 0.25], [-1.2, -0.5]],
        reject_limit=1.5,
    )

    assert projected.tolist() == [[1.0, 0.25], [-1.0, -0.5]]
    projected_release = (projected[0, 0] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    assert projected_release == pytest.approx(1.0 + 1e-6)


@pytest.mark.parametrize("value", [1.500001, float("nan")])
def test_gripper_latent_projection_rejects_invalid_model_output(value):
    with pytest.raises(ValueError, match="gripper latent"):
        project_gripper_quantile_latent([value], reject_limit=1.5)
