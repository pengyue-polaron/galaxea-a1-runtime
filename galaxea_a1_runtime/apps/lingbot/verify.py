"""Read-only verification for a composed LingBot deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

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
from embodied_ops.artifacts import file_sha256
from galaxea_a1_runtime.models.backend import verify_backend_environment
from galaxea_a1_runtime.models.store import validate_artifact


TrainingProvenance = Literal[
    "declared-code-revision",
    "embedded-inference-config",
    "uncommitted-training-source",
]


def verify_deployment(config: LingBotConfig) -> None:
    policy = config.policy_server
    if not policy.deployment_ready:
        raise RuntimeError("LingBot deployment refuses deployment.ready=false")
    verify_backend_environment(policy.backend)
    artifact = validate_artifact(policy.model, verify_hashes=True)
    provenance = validate_training_summary(config, artifact.root)
    if message := training_provenance_warning(provenance):
        warning(message)
    success(
        "LingBot deployment verified: "
        f"source={policy.backend.source.revision} model={policy.model.model_id} "
        f"revision={policy.model.source.revision} files={artifact.files} "
        f"manifest={artifact.manifest_sha256}"
    )


def validate_training_summary(
    config: LingBotConfig, artifact_root: Path
) -> TrainingProvenance:
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
    expected_repository = policy.backend.source.repository.removesuffix(".git")
    if code_repository is not None and code_revision is None:
        starting_revision = summary.get("starting_code_revision")
        valid_starting_revision = isinstance(starting_revision, str) and (
            len(starting_revision) == 40
            and all(character in "0123456789abcdef" for character in starting_revision)
        )
        expected_uncommitted = {
            "code_repository": expected_repository,
            "training_worktree_had_uncommitted_changes": True,
            "exact_training_files_included": True,
        }
        mismatched_uncommitted = {
            key: (summary.get(key), value)
            for key, value in expected_uncommitted.items()
            if summary.get(key) != value
        }
        if not valid_starting_revision:
            mismatched_uncommitted["starting_code_revision"] = (
                starting_revision,
                "40-character lowercase Git revision",
            )
        if mismatched_uncommitted:
            raise ValueError(
                "LingBot uncommitted training provenance mismatch: "
                f"{mismatched_uncommitted}"
            )
        _validate_embedded_inference_config(policy, artifact_root)
        return "uncommitted-training-source"
    if code_repository is None or code_revision is None:
        raise ValueError(
            "LingBot training summary must declare both code_repository and "
            "code_revision, or neither"
        )
    expected_code = {
        "code_repository": expected_repository,
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


def training_provenance_warning(provenance: TrainingProvenance) -> str | None:
    if provenance == "embedded-inference-config":
        return (
            "Training summary has no code revision; compatibility was verified "
            "by matching its embedded inference config to the pinned backend."
        )
    if provenance == "uncommitted-training-source":
        return (
            "Training summary records an uncommitted training worktree; its "
            "starting revision and exact training files are retained, and "
            "inference compatibility was verified against the pinned backend."
        )
    return None


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
