"""Read-only verification for a composed LingBot deployment."""

from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success
from galaxea_a1_runtime.models.backend import verify_backend_environment
from galaxea_a1_runtime.models.store import validate_artifact


def verify_deployment(config: LingBotConfig) -> None:
    policy = config.policy_server
    if not policy.deployment_ready:
        raise RuntimeError("LingBot deployment refuses deployment.ready=false")
    verify_backend_environment(policy.backend)
    artifact = validate_artifact(policy.model, verify_hashes=True)
    validate_training_summary(config, artifact.root)
    success(
        "LingBot deployment verified: "
        f"source={policy.backend.source.revision} model={policy.model.model_id} "
        f"revision={policy.model.source.revision} files={artifact.files} "
        f"manifest={artifact.manifest_sha256}"
    )


def validate_training_summary(config: LingBotConfig, artifact_root: Path) -> None:
    policy = config.policy_server
    summary = json.loads((artifact_root / "training_summary.json").read_text())
    if not isinstance(summary, dict):
        raise ValueError("LingBot training summary must be a JSON object")
    expected = {
        "checkpoint_step": policy.model.checkpoint_step,
        "code_repository": policy.backend.source.repository.removesuffix(".git"),
        "code_revision": policy.backend.source.revision,
        "source_action_dimension": len(policy.action_channel_ids),
        "model_action_dimension": policy.model_action_dim,
        "used_action_channel_ids": list(policy.action_channel_ids),
        "includes_optimizer_state": False,
    }
    mismatched = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatched:
        raise ValueError(f"LingBot training summary contract mismatch: {mismatched}")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(config_path, repo_root=repo_root)
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("LingBot config does not belong to --repo-root")
    verify_deployment(config)
    return 0
