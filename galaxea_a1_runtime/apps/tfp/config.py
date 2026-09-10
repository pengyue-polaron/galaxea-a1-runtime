"""Compose the TFP-Ultra source, checkpoint, task, and A1 System owners."""

from __future__ import annotations

import sys
from pathlib import Path

from embodied_ops import load_task_catalog

from galaxea_a1_runtime.apps.tfp.config_schema import (
    TFPConfig,
    TFPDiagnosticsConfig,
    TFPEngineConfig,
    TFPExecutionConfig,
    TFPModelConfig,
    TFPObservationConfig,
    TFPResetConfig,
)
from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    hex_digest,
    identifier,
    integer,
    load_toml,
    referenced_config,
    repo_path,
    require_exact_keys,
    required_table,
    string,
    string_tuple,
)
from galaxea_a1_runtime.configuration.paths import TFP_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.models.backend import CodeBackendConfig, parse_code_backend
from galaxea_a1_runtime.schema import (
    FRONT_IMAGE_FEATURE_KEY,
    JOINT_ACTION_NAMES_RAD,
    WRIST_IMAGE_FEATURE_KEY,
    camera_specs_from_system,
)


def default_config_path(repo_root: Path) -> Path:
    return repo_root / TFP_CONFIG


def load_tfp_config(path: Path, *, repo_root: Path | None = None) -> TFPConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "system",
            "backend",
            "tasks",
            "model",
            "deployment",
            "observations",
            "reset",
            "execution",
            "diagnostics",
        },
        label="TFP deployment config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    backend, engine = _load_backend(
        referenced_config(data, repo_root, key="backend"), repo_root
    )
    task_catalog = load_task_catalog(
        referenced_config(data, repo_root, key="tasks"), repo_root=repo_root
    )

    model = required_table(data, "model")
    deployment = required_table(data, "deployment")
    observations = required_table(data, "observations")
    reset = required_table(data, "reset")
    execution = required_table(data, "execution")
    diagnostics = required_table(data, "diagnostics")
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
            "action_horizon",
            "actions_per_query",
            "max_training_episode_actions",
        },
        label="TFP model",
    )
    require_exact_keys(
        deployment,
        required={"id", "ready", "task_id"},
        label="TFP deployment",
    )
    require_exact_keys(
        observations,
        required={"front_key", "wrist_key"},
        label="TFP observations",
    )
    require_exact_keys(
        reset,
        required={"before_run", "config"},
        label="TFP reset",
    )
    require_exact_keys(
        execution,
        required={
            "execute",
            "max_actions",
            "exec_rate",
            "observation_timeout_s",
            "print_actions",
        },
        label="TFP execution",
    )
    require_exact_keys(
        diagnostics,
        required={
            "enabled",
            "output_dir",
            "print_every_queries",
            "include_belief_vectors",
        },
        label="TFP diagnostics",
    )
    task_id = identifier(string(deployment, "task_id"), label="deployment.task_id")
    task_catalog.task(task_id)
    config = TFPConfig(
        path=path,
        repo_root=repo_root,
        deployment_id=identifier(string(deployment, "id"), label="deployment.id"),
        deployment_ready=boolean(deployment, "ready"),
        task_id=task_id,
        system=system,
        backend=backend,
        engine=engine,
        task_catalog=task_catalog,
        model=TFPModelConfig(
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
            action_names=string_tuple(
                model, "action_names", len(JOINT_ACTION_NAMES_RAD)
            ),
            action_horizon=integer(model, "action_horizon"),
            actions_per_query=integer(model, "actions_per_query"),
            max_training_episode_actions=integer(model, "max_training_episode_actions"),
        ),
        observations=TFPObservationConfig(
            front_key=string(observations, "front_key"),
            wrist_key=string(observations, "wrist_key"),
        ),
        reset=TFPResetConfig(
            before_run=boolean(reset, "before_run"),
            config=repo_path(repo_root, string(reset, "config")),
        ),
        execution=TFPExecutionConfig(
            execute=boolean(execution, "execute"),
            max_actions=integer(execution, "max_actions"),
            exec_rate=floating(execution, "exec_rate"),
            observation_timeout_s=floating(execution, "observation_timeout_s"),
            print_actions=boolean(execution, "print_actions"),
        ),
        diagnostics=TFPDiagnosticsConfig(
            enabled=boolean(diagnostics, "enabled"),
            output_dir=repo_path(repo_root, string(diagnostics, "output_dir")),
            print_every_queries=integer(diagnostics, "print_every_queries"),
            include_belief_vectors=boolean(diagnostics, "include_belief_vectors"),
        ),
    )
    validate_tfp_config(config)
    return config


def _load_backend(
    path: Path, repo_root: Path
) -> tuple[CodeBackendConfig, TFPEngineConfig]:
    _, _, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"backend", "source", "environment", "engine"},
        label="TFP backend config",
    )
    backend = parse_code_backend(
        backend=required_table(data, "backend"),
        source=required_table(data, "source"),
        environment=required_table(data, "environment"),
        repo_root=repo_root,
    )
    engine = required_table(data, "engine")
    package_keys = (
        "torch_version",
        "torchvision_version",
        "diffusers_version",
        "ncps_version",
    )
    require_exact_keys(
        engine,
        required={"device", "seed", *package_keys},
        label="TFP engine",
    )
    return backend, TFPEngineConfig(
        device=string(engine, "device"),
        seed=integer(engine, "seed"),
        package_versions=tuple(
            (key.removesuffix("_version"), string(engine, key)) for key in package_keys
        ),
    )


def validate_tfp_config(config: TFPConfig) -> None:
    if config.backend.adapter != "tfp_ultra":
        raise ValueError("TFP backend.adapter must be 'tfp_ultra'")
    if config.engine.device != "cuda":
        raise ValueError("TFP engine.device must be 'cuda'")
    if config.engine.seed < 0:
        raise ValueError("TFP engine.seed must be non-negative")
    if config.model.checkpoint_format != "training-state":
        raise ValueError("TFP checkpoint_format must be 'training-state'")
    if not config.model.checkpoint.is_relative_to(
        (config.repo_root / "models/checkpoints").resolve()
    ):
        raise ValueError("TFP checkpoint must remain under models/checkpoints/")
    if not config.model.metadata.is_relative_to(config.backend.source.checkout):
        raise ValueError("TFP metadata must remain inside the pinned source checkout")
    if config.model.action_names != JOINT_ACTION_NAMES_RAD:
        raise ValueError(
            "TFP action_names do not match the canonical A1 joint contract"
        )
    if (
        min(
            config.model.checkpoint_step,
            config.model.checkpoint_size,
            config.model.action_horizon,
            config.model.actions_per_query,
            config.model.max_training_episode_actions,
        )
        <= 0
    ):
        raise ValueError(
            "TFP model dimensions, checkpoint identity, and budget must be positive"
        )
    if config.model.actions_per_query > config.model.action_horizon:
        raise ValueError("TFP actions_per_query exceeds action_horizon")
    if config.execution.execute and not config.deployment_ready:
        raise ValueError("TFP execution.execute requires deployment.ready=true")
    if config.execution.max_actions < 0:
        raise ValueError("TFP max_actions must be >= 0 (zero means unlimited)")
    if config.execution.exec_rate <= 0 or config.execution.observation_timeout_s <= 0:
        raise ValueError("TFP execution rate and observation timeout must be positive")
    if (
        config.execution.observation_timeout_s
        >= config.system.robot_service.command_timeout_s
    ):
        raise ValueError(
            "TFP observation timeout must be below the robot command timeout"
        )
    if config.execution.exec_rate != config.system.cameras.front.fps:
        raise ValueError("TFP execution rate must match the front camera FPS")
    if config.diagnostics.print_every_queries <= 0:
        raise ValueError("TFP diagnostics.print_every_queries must be positive")
    if not config.diagnostics.output_dir.is_relative_to(
        (config.repo_root / "outputs").resolve()
    ):
        raise ValueError("TFP diagnostics output must remain under outputs/")
    from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose

    load_a1_home_pose(
        config.reset.config,
        system=config.system,
        repo_root=config.repo_root,
    )
    if (config.observations.front_key, config.observations.wrist_key) != (
        FRONT_IMAGE_FEATURE_KEY,
        WRIST_IMAGE_FEATURE_KEY,
    ):
        raise ValueError("TFP observation keys do not match the training contract")
    shapes = tuple(
        (camera.height, camera.width, camera.channels)
        for camera in camera_specs_from_system(config.system)
    )
    if shapes != ((480, 480, 3), (480, 640, 3)):
        raise ValueError(f"TFP camera shapes differ from training: {shapes}")


def main(argv: list[str] | None = None) -> int:
    from galaxea_a1_runtime.apps.tfp.config_runtime import bash_config
    from galaxea_a1_runtime.configuration.cli import run_config_renderer

    return run_config_renderer(
        argv,
        description="Read the composed A1 TFP-Ultra deployment config.",
        default_config=TFP_CONFIG,
        load_config=load_tfp_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
