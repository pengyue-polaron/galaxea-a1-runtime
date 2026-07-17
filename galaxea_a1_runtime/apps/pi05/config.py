"""Compose the OpenPI backend, immutable model, deployment, and System owners."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import Any

from galaxea_a1_runtime.apps.pi05.config_schema import (
    Pi05Config,
    Pi05EngineConfig,
    Pi05ExecutionConfig,
    Pi05ModelContract,
    Pi05ObservationConfig,
    Pi05ServerConfig,
    Pi05ServoConfig,
    Pi05SessionConfig,
    PoseMode,
)
from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    load_toml,
    referenced_config,
    require_exact_keys,
    repo_path,
    required_table,
    string,
    text,
)
from galaxea_a1_runtime.configuration.paths import PI05_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.models.backend import CodeBackendConfig, parse_code_backend
from galaxea_a1_runtime.models.config import ModelArtifactConfig, load_model_config
from galaxea_a1_runtime.schema import EEF_ACTION_NAMES, EEF_DATASET_STATE_NAMES


DEFAULT_PI05_CONFIG = PI05_CONFIG


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_PI05_CONFIG


def load_pi05_config(path: Path, *, repo_root: Path | None = None) -> Pi05Config:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={
            "system",
            "backend",
            "model",
            "deployment",
            "session",
            "server",
            "observations",
            "execution",
            "action",
        },
        label="pi0.5 deployment config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)
    backend, engine = _load_backend(
        _config_reference(data, "backend", repo_root), repo_root
    )
    model = load_model_config(
        _config_reference(data, "model", repo_root), repo_root=repo_root
    )
    if backend.adapter != "openpi_pi05" or model.backend != backend.backend_id:
        raise ValueError(
            "pi0.5 deployment backend/model mismatch: "
            f"adapter={backend.adapter!r}, backend={backend.backend_id!r}, "
            f"model.backend={model.backend!r}"
        )
    if model.artifact_format != "orbax-ocdbt":
        raise ValueError("pi0.5 model artifact_format must be 'orbax-ocdbt'")
    model_contract = _load_model_contract(model)

    deployment = required_table(data, "deployment")
    session = required_table(data, "session")
    server = required_table(data, "server")
    observations = required_table(data, "observations")
    execution = required_table(data, "execution")
    action = required_table(data, "action")
    require_exact_keys(deployment, required={"id", "ready"}, label="pi0.5 deployment")
    require_exact_keys(
        session,
        required={"tmux", "model_tmux", "startup_timeout_s"},
        label="pi0.5 session",
    )
    require_exact_keys(
        server,
        required={"host", "port", "connect_timeout_s", "close_timeout_s", "prompt"},
        label="pi0.5 server",
    )
    require_exact_keys(
        observations, required={"front_key", "wrist_key"}, label="pi0.5 observations"
    )
    require_exact_keys(
        execution,
        required={
            "execute",
            "step_mode",
            "step_actions",
            "max_model_calls",
            "execute_actions_per_inference",
            "exec_rate",
            "print_actions",
            "review_deadband_m",
        },
        label="pi0.5 execution",
    )
    require_exact_keys(
        action,
        required={
            "servo_gain",
            "servo_max_extra_m",
            "servo_settle_s",
            "servo_tolerance_m",
            "servo_corrections",
        },
        label="pi0.5 action",
    )
    config = Pi05Config(
        path=path,
        deployment_id=_safe_id(string(deployment, "id"), label="deployment.id"),
        deployment_ready=boolean(deployment, "ready"),
        system=system,
        backend=backend,
        engine=engine,
        model=model,
        model_contract=model_contract,
        session=Pi05SessionConfig(
            tmux=string(session, "tmux"),
            model_tmux=string(session, "model_tmux"),
            startup_timeout_s=floating(session, "startup_timeout_s"),
        ),
        server=Pi05ServerConfig(
            host=string(server, "host"),
            port=integer(server, "port"),
            connect_timeout_s=floating(server, "connect_timeout_s"),
            close_timeout_s=floating(server, "close_timeout_s"),
            prompt=text(server, "prompt").strip(),
        ),
        observations=Pi05ObservationConfig(
            front_key=string(observations, "front_key"),
            wrist_key=string(observations, "wrist_key"),
        ),
        execution=Pi05ExecutionConfig(
            execute=boolean(execution, "execute"),
            step_mode=boolean(execution, "step_mode"),
            step_actions=boolean(execution, "step_actions"),
            max_model_calls=integer(execution, "max_model_calls"),
            execute_actions_per_inference=integer(
                execution, "execute_actions_per_inference"
            ),
            exec_rate=floating(execution, "exec_rate"),
            print_actions=boolean(execution, "print_actions"),
            review_deadband_m=floating(execution, "review_deadband_m"),
        ),
        servo=Pi05ServoConfig(
            gain=floating(action, "servo_gain"),
            max_extra_m=floating(action, "servo_max_extra_m"),
            settle_s=floating(action, "servo_settle_s"),
            tolerance_m=floating(action, "servo_tolerance_m"),
            corrections=integer(action, "servo_corrections"),
        ),
    )
    validate_pi05_config(config)
    return config


def _load_backend(
    path: Path, repo_root: Path
) -> tuple[CodeBackendConfig, Pi05EngineConfig]:
    _, _, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"backend", "source", "environment", "engine"},
        label="pi0.5 backend config",
    )
    backend = parse_code_backend(
        backend=required_table(data, "backend"),
        source=required_table(data, "source"),
        environment=required_table(data, "environment"),
        repo_root=repo_root,
    )
    engine = required_table(data, "engine")
    require_exact_keys(
        engine,
        required={"jax_platform", "xla_memory_fraction", "seed", "sampling_steps"},
        label="pi0.5 engine",
    )
    platform = string(engine, "jax_platform")
    if platform != "cuda":
        raise ValueError("pi0.5 engine.jax_platform must be 'cuda'")
    return backend, Pi05EngineConfig(
        jax_platform=platform,
        xla_memory_fraction=floating(engine, "xla_memory_fraction"),
        seed=integer(engine, "seed"),
        sampling_steps=integer(engine, "sampling_steps"),
    )


def _load_model_contract(model: ModelArtifactConfig) -> Pi05ModelContract:
    _, _, data = load_toml(model.contract, repo_root=model.repo_root)
    require_exact_keys(data, required={"openpi"}, label="pi0.5 model contract")
    contract = required_table(data, "openpi")
    require_exact_keys(
        contract,
        required={
            "train_config",
            "checkpoint_format",
            "parameter_set",
            "norm_stats_path",
            "pose_mode",
            "action_horizon",
            "state_dim",
            "source_action_dim",
            "model_action_dim",
        },
        label="pi0.5 model contract",
    )
    norm_stats = _artifact_relative_path(string(contract, "norm_stats_path"))
    pose_mode = _pose_mode(string(contract, "pose_mode"))
    return Pi05ModelContract(
        train_config=string(contract, "train_config"),
        checkpoint_format=string(contract, "checkpoint_format"),
        parameter_set=string(contract, "parameter_set"),
        norm_stats_path=model.artifact_root.joinpath(*norm_stats.parts),
        pose_mode=pose_mode,
        action_horizon=integer(contract, "action_horizon"),
        state_dim=integer(contract, "state_dim"),
        source_action_dim=integer(contract, "source_action_dim"),
        model_action_dim=integer(contract, "model_action_dim"),
    )


def validate_pi05_config(config: Pi05Config) -> None:
    if not 1 <= config.server.port <= 65535:
        raise ValueError("pi0.5 server.port must be in [1, 65535]")
    if (
        min(
            config.server.connect_timeout_s,
            config.server.close_timeout_s,
            config.session.startup_timeout_s,
        )
        <= 0
    ):
        raise ValueError("pi0.5 server and startup timeouts must be positive")
    if not 0 < config.engine.xla_memory_fraction <= 1:
        raise ValueError("pi0.5 xla_memory_fraction must be in (0, 1]")
    if config.engine.seed != 0:
        raise ValueError("the pinned OpenPI policy constructor requires engine.seed=0")
    if config.engine.sampling_steps <= 0:
        raise ValueError("pi0.5 sampling_steps must be positive")
    contract = config.model_contract
    if contract.checkpoint_format != config.model.artifact_format:
        raise ValueError("pi0.5 checkpoint format does not match the model descriptor")
    if contract.parameter_set != "ema_params":
        raise ValueError("pi0.5 deployment accepts only the published EMA parameters")
    if contract.state_dim != len(EEF_DATASET_STATE_NAMES):
        raise ValueError(
            "pi0.5 state_dim does not match the shared A1 EEF state schema"
        )
    if contract.source_action_dim != len(EEF_ACTION_NAMES):
        raise ValueError(
            "pi0.5 source_action_dim does not match the shared EEF action schema"
        )
    if min(contract.action_horizon, contract.model_action_dim) <= 0:
        raise ValueError("pi0.5 model action dimensions must be positive")
    manifest_paths = {item.path.as_posix() for item in config.model.manifest.files}
    required = {
        "checkpoint_manifest.json",
        "training_summary.json",
        "params/_METADATA",
        config.model_contract.norm_stats_path.relative_to(
            config.model.artifact_root
        ).as_posix(),
    }
    missing = sorted(required - manifest_paths)
    if missing:
        raise ValueError(f"pi0.5 model manifest is missing contract files: {missing}")
    if config.deployment_ready and not config.server.prompt:
        raise ValueError("deployment-ready pi0.5 config requires a real prompt")
    execution = config.execution
    if execution.execute and not config.deployment_ready:
        raise ValueError("pi0.5 execution.execute requires deployment.ready=true")
    if execution.max_model_calls < 0:
        raise ValueError("pi0.5 max_model_calls must be >= 0")
    if not 1 <= execution.execute_actions_per_inference <= contract.action_horizon:
        raise ValueError("pi0.5 execute_actions_per_inference exceeds action_horizon")
    if execution.exec_rate <= 0 or execution.review_deadband_m < 0:
        raise ValueError(
            "pi0.5 execution rate must be positive and deadband non-negative"
        )
    if config.system.cameras.front.backend != "realsense":
        raise ValueError("pi0.5 front camera must use the RealSense backend")
    if config.servo.gain <= 0 or config.servo.tolerance_m <= 0:
        raise ValueError("pi0.5 servo gain and tolerance must be positive")
    if config.servo.max_extra_m < 0 or config.servo.settle_s < 0:
        raise ValueError("pi0.5 servo max_extra_m and settle_s must be non-negative")
    if config.servo.corrections < 0:
        raise ValueError("pi0.5 servo.corrections must be >= 0")
    if config.servo.corrections and config.servo.settle_s <= 0:
        raise ValueError("pi0.5 servo corrections require a positive settle time")


def _config_reference(data: dict[str, Any], key: str, repo_root: Path) -> Path:
    table = required_table(data, key)
    require_exact_keys(table, required={"config"}, label=f"{key} reference")
    return repo_path(repo_root, string(table, "config"))


def _artifact_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"invalid pi0.5 artifact-relative path: {value!r}")
    return path


def _pose_mode(value: str) -> PoseMode:
    if value not in {"absolute", "episode-relative"}:
        raise ValueError(f"unsupported pi0.5 pose_mode: {value!r}")
    return value


def _safe_id(value: str, *, label: str) -> str:
    if not value or any(
        not (character.isalnum() or character in {"-", "_", "."}) for character in value
    ):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    from galaxea_a1_runtime.configuration.cli import run_config_renderer
    from galaxea_a1_runtime.apps.pi05.config_runtime import bash_config

    return run_config_renderer(
        argv,
        description="Read the composed A1 pi0.5 deployment config.",
        default_config=DEFAULT_PI05_CONFIG,
        load_config=load_pi05_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    sys.exit(main())
