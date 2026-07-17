"""Read-only verification for a composed OpenPI pi0.5 deployment."""

from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.apps.pi05.config import default_config_path, load_pi05_config
from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success
from galaxea_a1_runtime.models.backend import verify_backend_environment
from galaxea_a1_runtime.models.store import validate_artifact


def verify_deployment(config: Pi05Config) -> None:
    if not config.deployment_ready:
        raise RuntimeError("pi0.5 deployment refuses deployment.ready=false")
    verify_backend_environment(config.backend)
    artifact = validate_artifact(config.model, verify_hashes=True)
    validate_training_summary(config, artifact.root)
    success(
        "OpenPI pi0.5 deployment verified: "
        f"source={config.backend.source.revision} model={config.model.model_id} "
        f"step={config.model.checkpoint_step} files={artifact.files} "
        f"manifest={artifact.manifest_sha256}"
    )


def validate_training_summary(config: Pi05Config, artifact_root: Path) -> None:
    summary = json.loads((artifact_root / "training_summary.json").read_text())
    if not isinstance(summary, dict):
        raise ValueError("pi0.5 training summary must be a JSON object")
    norm_stats = config.model_contract.norm_stats_path.relative_to(
        config.model.artifact_root
    )
    inference_payload_bytes = sum(
        item.size
        for item in config.model.manifest.files
        if item.path.parts[0] == "params" or item.path == norm_stats
    )
    expected_summary = {
        "checkpoint_step": config.model.checkpoint_step,
        "checkpoint_tag": config.model.source.revision_label,
        "checkpoint_parameter_set": config.model_contract.parameter_set,
        "checkpoint_format": "Orbax OCDBT",
        "inference_payload_bytes": inference_payload_bytes,
        "code_repository": config.backend.source.repository.removesuffix(".git"),
        "code_revision": config.backend.source.revision,
        "action_horizon": config.model_contract.action_horizon,
        "source_action_dimension": config.model_contract.source_action_dim,
        "model_action_dimension": config.model_contract.model_action_dim,
        "includes_optimizer_state": False,
    }
    mismatched = {
        key: (summary.get(key), value)
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if mismatched:
        raise ValueError(f"pi0.5 training summary contract mismatch: {mismatched}")
    checkpoint_manifest = json.loads(
        (artifact_root / "checkpoint_manifest.json").read_text()
    )
    if not isinstance(checkpoint_manifest, dict):
        raise ValueError("pi0.5 checkpoint manifest must be a JSON object")
    defaults = [
        item
        for item in checkpoint_manifest.get("checkpoints", [])
        if isinstance(item, dict) and item.get("default_revision") is True
    ]
    expected_default = {
        "tag": config.model.source.revision_label,
        "checkpoint_step": config.model.checkpoint_step,
        "inference_payload_bytes": inference_payload_bytes,
    }
    if (
        checkpoint_manifest.get("format") != config.model_contract.checkpoint_format
        or checkpoint_manifest.get("published_parameter_set")
        != config.model_contract.parameter_set
        or checkpoint_manifest.get("includes_optimizer_state") is not False
        or len(defaults) != 1
        or any(defaults[0].get(key) != value for key, value in expected_default.items())
    ):
        raise ValueError(
            "pi0.5 checkpoint manifest does not identify the selected final revision"
        )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_pi05_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("pi0.5 config does not belong to --repo-root")
    verify_deployment(config)
    return 0
