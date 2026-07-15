"""Git-tracked ACT joint-state runtime configuration."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    load_toml,
    referenced_config,
    require_exact_keys,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.configuration.cli import run_config_renderer
from galaxea_a1_runtime.configuration.paths import ACT_CONFIG
from galaxea_a1_runtime.apps.act.config_runtime import bash_config
from galaxea_a1_runtime.apps.act.config_schema import (
    ActConfig,
    ActExecutionConfig,
    ActPolicyConfig,
    ActRuntimeConfig,
    ActSessionConfig,
)

DEFAULT_ACT_CONFIG = ACT_CONFIG

__all__ = ["bash_config", "load_act_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ACT_CONFIG


def load_act_config(path: Path, *, repo_root: Path | None = None) -> ActConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"system", "session", "runtime", "policy", "execution"},
        label="ACT config",
    )
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)

    session = _required_table(data, "session")
    runtime = _required_table(data, "runtime")
    policy = _required_table(data, "policy")
    execution = _required_table(data, "execution")
    require_exact_keys(session, required={"tmux"}, label="session")
    require_exact_keys(runtime, required={"prefix"}, label="runtime")
    require_exact_keys(
        policy,
        required={
            "checkpoint",
            "device",
            "disable_backbone_download",
            "deployment_ready",
        },
        label="policy",
    )
    require_exact_keys(
        execution,
        required={
            "execute",
            "step_mode",
            "execute_steps_per_inference",
            "control_hz",
            "max_model_calls",
            "print_actions",
            "preview_steps",
        },
        label="execution",
    )
    config = ActConfig(
        path=path,
        system=system,
        session=ActSessionConfig(tmux=_string(session, "tmux")),
        runtime=ActRuntimeConfig(
            prefix=_string(runtime, "prefix"),
        ),
        policy=ActPolicyConfig(
            checkpoint=_repo_path(repo_root, _string(policy, "checkpoint")),
            device=_string(policy, "device"),
            disable_backbone_download=boolean(policy, "disable_backbone_download"),
            deployment_ready=boolean(policy, "deployment_ready"),
        ),
        execution=ActExecutionConfig(
            execute=boolean(execution, "execute"),
            step_mode=boolean(execution, "step_mode"),
            execute_steps_per_inference=integer(
                execution, "execute_steps_per_inference"
            ),
            control_hz=floating(execution, "control_hz"),
            max_model_calls=integer(execution, "max_model_calls"),
            print_actions=boolean(execution, "print_actions"),
            preview_steps=integer(execution, "preview_steps"),
        ),
    )
    validate_act_config(config)
    return config


def validate_act_config(config: ActConfig) -> None:
    front = config.system.cameras.front
    if front.backend != "realsense":
        raise ValueError("ACT AgentView camera must use the realsense backend")
    if config.execution.execute_steps_per_inference <= 0:
        raise ValueError("execution.execute_steps_per_inference must be positive")
    if config.execution.control_hz <= 0:
        raise ValueError("execution.control_hz must be positive")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.preview_steps <= 0:
        raise ValueError("execution.preview_steps must be positive")
    if config.execution.execute and not config.policy.deployment_ready:
        raise ValueError("execution.execute requires policy.deployment_ready=true")


def main(argv: list[str] | None = None) -> int:
    return run_config_renderer(
        argv,
        description="Read the tracked A1 ACT deployment config.",
        default_config=DEFAULT_ACT_CONFIG,
        load_config=load_act_config,
        render_shell=bash_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
