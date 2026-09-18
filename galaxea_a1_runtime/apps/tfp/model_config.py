"""Immutable identity of the transferred TFP checkpoint."""

from pathlib import Path

from galaxea_a1_runtime.apps.tfp.config_schema import TFPModelConfig
from galaxea_a1_runtime.configuration.base import (
    hex_digest,
    identifier,
    integer,
    load_toml,
    repo_path,
    require_exact_keys,
    required_table,
    string,
    string_tuple,
)
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES_RAD


def load_tfp_model_config(path: Path, *, repo_root: Path) -> TFPModelConfig:
    _, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(data, required={"model"}, label="TFP model descriptor")
    model = required_table(data, "model")
    require_exact_keys(
        model,
        required={
            "id",
            "checkpoint",
            "checkpoint_format",
            "checkpoint_step",
            "checkpoint_size",
            "checkpoint_sha256",
            "metadata",
            "metadata_info_sha256",
            "metadata_stats_sha256",
            "dataset_repo_id",
            "dataset_revision",
            "action_names",
        },
        label="TFP model",
    )
    config = TFPModelConfig(
        model_id=identifier(string(model, "id"), label="model.id"),
        checkpoint=repo_path(repo_root, string(model, "checkpoint")),
        checkpoint_format=string(model, "checkpoint_format"),
        checkpoint_step=integer(model, "checkpoint_step"),
        checkpoint_size=integer(model, "checkpoint_size"),
        checkpoint_sha256=hex_digest(
            string(model, "checkpoint_sha256"), 64, label="model.checkpoint_sha256"
        ),
        metadata=repo_path(repo_root, string(model, "metadata")),
        metadata_info_sha256=hex_digest(
            string(model, "metadata_info_sha256"),
            64,
            label="model.metadata_info_sha256",
        ),
        metadata_stats_sha256=hex_digest(
            string(model, "metadata_stats_sha256"),
            64,
            label="model.metadata_stats_sha256",
        ),
        dataset_repo_id=string(model, "dataset_repo_id"),
        dataset_revision=hex_digest(
            string(model, "dataset_revision"), 40, label="model.dataset_revision"
        ),
        action_names=string_tuple(model, "action_names", len(JOINT_ACTION_NAMES_RAD)),
    )
    if config.checkpoint_format != "training-state":
        raise ValueError("TFP checkpoint_format must be 'training-state'")
    if not config.checkpoint.is_relative_to(
        (repo_root / "models/checkpoints").resolve()
    ):
        raise ValueError("TFP checkpoint must remain under models/checkpoints/")
    if config.action_names != JOINT_ACTION_NAMES_RAD:
        raise ValueError(
            "TFP action_names do not match the canonical A1 joint contract"
        )
    if min(config.checkpoint_step, config.checkpoint_size) <= 0:
        raise ValueError("TFP checkpoint step and size must be positive")
    return config
