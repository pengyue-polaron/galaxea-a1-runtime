from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
from galaxea_a1_runtime.apps.pi05 import probe as probe_module
from galaxea_a1_runtime.apps.pi05.protocol import (
    PROTOCOL_VERSION,
    server_metadata,
    validate_server_metadata,
)
from galaxea_a1_runtime.apps.pi05.verify import validate_training_summary
from galaxea_a1_runtime.schema import A1_STATE_NAMES


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/deployments/pi05/fruit_placement_eef.toml"
MODEL = REPO / "configs/models/pi05/fruit_placement_eef.toml"
CONTRACT = REPO / "configs/models/pi05/fruit_placement_eef.contract.toml"


def _deployment_copy(tmp_path: Path, contract_text: str) -> Path:
    contract_path = tmp_path / "contract.toml"
    contract_path.write_text(contract_text)
    descriptor = MODEL.read_text().replace(
        'contract = "configs/models/pi05/fruit_placement_eef.contract.toml"',
        f'contract = "{contract_path}"',
    )
    model_path = tmp_path / "model.toml"
    model_path.write_text(descriptor)
    deployment = CONFIG.read_text().replace(
        'config = "configs/models/pi05/fruit_placement_eef.toml"',
        f'config = "{model_path}"',
    )
    path = tmp_path / "deployment.toml"
    path.write_text(deployment)
    return path


def test_pi05_deployment_pins_final_checkpoint_and_shared_contracts():
    config = load_pi05_config(CONFIG, repo_root=REPO)

    assert config.model.model_id == "openpi_pi05/a1_fruit_placement_eef"
    assert config.model.checkpoint_step == 14999
    assert config.model.source.revision_label == "step-14999"
    assert config.model.source.revision == "e1a3e53832ce99edc188fb01e5ec303ac305d552"
    assert config.backend.source.revision == "55f9842731bfa4aed851d068e0bb5693f6289fe9"
    assert config.model_contract.action_horizon == 10
    assert config.model_contract.source_action_dim == 8
    assert config.model_contract.state_dim == 14
    assert config.model_contract.pose_mode == "episode-relative"
    assert config.task_catalog.path == (
        REPO / "configs/tasks/fruit_placement/catalog.json"
    )
    assert config.task_catalog.task("lemon_bowl").distribution == "ood"
    assert config.execution.execute is True
    assert config.execution.step_mode is False
    assert config.execution.step_actions is False
    assert config.execution.max_model_calls == 53
    assert config.execution.execute_actions_per_inference == 10
    assert config.deployment_ready is True


def test_pi05_config_rejects_unknown_keys(tmp_path):
    path = _deployment_copy(tmp_path, CONTRACT.read_text())
    path.write_text(path.read_text() + "\n[unknown]\nvalue = true\n")

    with pytest.raises(ValueError, match="unknown"):
        load_pi05_config(path, repo_root=REPO)


def test_pi05_probe_rejects_config_from_another_repository(tmp_path, monkeypatch):
    foreign = tmp_path / "foreign"
    (foreign / "galaxea_a1_runtime").mkdir(parents=True)
    (foreign / "pyproject.toml").write_text("[project]\nname = 'foreign'\n")
    config_path = foreign / "deployment.toml"
    config_path.touch()
    monkeypatch.setattr(
        probe_module,
        "load_pi05_config",
        lambda *_args, **_kwargs: SimpleNamespace(path=config_path),
    )

    with pytest.raises(ValueError, match="does not belong to --repo-root"):
        probe_module.main(["--repo-root", str(REPO), "--config", str(config_path)])


def test_pi05_protocol_exhaustively_identifies_model_and_io_contract():
    config = load_pi05_config(CONFIG, repo_root=REPO)

    metadata = server_metadata(config)

    assert metadata["protocol"] == PROTOCOL_VERSION
    assert metadata["deployment_id"] == config.deployment_id
    assert metadata["environment"]["python_version"] == "3.11"
    assert metadata["checkpoint_step"] == 14999
    assert metadata["model_revision_label"] == "step-14999"
    assert metadata["task_catalog"] == config.task_catalog.protocol_contract()
    assert metadata["camera_shapes"] == [[480, 480, 3], [480, 640, 3]]
    assert metadata["state_shape"] == [14]
    assert metadata["state_names"] == list(A1_STATE_NAMES)
    assert metadata["action_shape"] == [10, 8]
    assert metadata["pose_mode"] == "episode-relative"
    assert len(metadata["contract_sha256"]) == 64

    drifted = dict(metadata)
    drifted["checkpoint_step"] = 10000
    with pytest.raises(RuntimeError, match="contract mismatch: checkpoint_step"):
        validate_server_metadata(drifted, metadata)


def test_pi05_rejects_checkpoint_contract_dimension_drift(tmp_path):
    contract = CONTRACT.read_text().replace("state_dim = 14", "state_dim = 13")
    path = _deployment_copy(tmp_path, contract)

    with pytest.raises(ValueError, match="state_dim.*shared A1 EEF"):
        load_pi05_config(path, repo_root=REPO)


def test_pi05_training_metadata_identifies_the_selected_final_checkpoint(tmp_path):
    config = load_pi05_config(CONFIG, repo_root=REPO)
    norm_stats = config.model_contract.norm_stats_path.relative_to(
        config.model.artifact_root
    )
    payload_bytes = sum(
        item.size
        for item in config.model.manifest.files
        if item.path.parts[0] == "params" or item.path == norm_stats
    )
    summary = {
        "checkpoint_step": config.model.checkpoint_step,
        "checkpoint_tag": config.model.source.revision_label,
        "checkpoint_parameter_set": config.model_contract.parameter_set,
        "checkpoint_format": "Orbax OCDBT",
        "inference_payload_bytes": payload_bytes,
        "code_repository": config.backend.source.repository.removesuffix(".git"),
        "code_revision": config.backend.source.revision,
        "action_horizon": config.model_contract.action_horizon,
        "source_action_dimension": config.model_contract.source_action_dim,
        "model_action_dimension": config.model_contract.model_action_dim,
        "includes_optimizer_state": False,
    }
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    checkpoint_manifest = {
        "format": config.model_contract.checkpoint_format,
        "published_parameter_set": config.model_contract.parameter_set,
        "includes_optimizer_state": False,
        "checkpoints": [
            {
                "tag": config.model.source.revision_label,
                "checkpoint_step": config.model.checkpoint_step,
                "inference_payload_bytes": payload_bytes,
                "default_revision": True,
            }
        ],
    }
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest))

    validate_training_summary(config, tmp_path)

    summary["checkpoint_step"] = 10000
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="training summary contract mismatch"):
        validate_training_summary(config, tmp_path)
