"""Read-only verification for a composed LingBot deployment."""

from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import (
    LingBotConfig,
    LingBotPolicyServerConfig,
)
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success, warning
from galaxea_a1_runtime.filesystem import file_sha256
from galaxea_a1_runtime.models.backend import verify_backend_environment
from galaxea_a1_runtime.models.store import validate_artifact


def verify_deployment(config: LingBotConfig) -> None:
    policy = config.policy_server
    if not policy.deployment_ready:
        raise RuntimeError("LingBot deployment refuses deployment.ready=false")
    verify_backend_environment(policy.backend)
    artifact = validate_artifact(policy.model, verify_hashes=True)
    provenance = validate_training_summary(config, artifact.root)
    if provenance == "embedded-inference-config":
        warning(
            "Training summary has no code revision; compatibility was verified "
            "by matching its embedded inference config to the pinned backend."
        )
    success(
        "LingBot deployment verified: "
        f"source={policy.backend.source.revision} model={policy.model.model_id} "
        f"revision={policy.model.source.revision} files={artifact.files} "
        f"manifest={artifact.manifest_sha256}"
    )


def validate_training_summary(config: LingBotConfig, artifact_root: Path) -> str:
    policy = config.policy_server
    summary = json.loads((artifact_root / "training_summary.json").read_text())
    if not isinstance(summary, dict):
        raise ValueError("LingBot training summary must be a JSON object")
    expected = {
        "checkpoint_step": policy.model.checkpoint_step,
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
    code_repository = summary.get("code_repository")
    code_revision = summary.get("code_revision")
    if code_repository is None and code_revision is None:
        _validate_embedded_inference_config(policy, artifact_root)
        return "embedded-inference-config"
    if code_repository is None or code_revision is None:
        raise ValueError(
            "LingBot training summary must declare both code_repository and "
            "code_revision, or neither"
        )
    expected_code = {
        "code_repository": policy.backend.source.repository.removesuffix(".git"),
        "code_revision": policy.backend.source.revision,
    }
    mismatched_code = {
        key: (summary.get(key), value)
        for key, value in expected_code.items()
        if summary.get(key) != value
    }
    if mismatched_code:
        raise ValueError(
            f"LingBot training summary contract mismatch: {mismatched_code}"
        )
    return "declared-code-revision"


def _validate_embedded_inference_config(
    policy: LingBotPolicyServerConfig,
    artifact_root: Path,
) -> None:
    filename = f"va_{policy.vendor_config}_cfg.py"
    embedded = artifact_root / "configs" / filename
    pinned = policy.backend.source.checkout / "wan_va" / "configs" / filename
    missing = [str(path) for path in (embedded, pinned) if not path.is_file()]
    if missing:
        raise ValueError(
            "LingBot training summary has no code provenance and its inference "
            f"config compatibility cannot be verified; missing: {missing}"
        )
    if file_sha256(embedded) != file_sha256(pinned):
        raise ValueError(
            "LingBot training summary has no code provenance and its embedded "
            "inference config does not match the pinned backend"
        )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(
        config_path,
        repo_root=repo_root,
        model_selector=args.model,
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("LingBot config does not belong to --repo-root")
    verify_deployment(config)
    return 0
