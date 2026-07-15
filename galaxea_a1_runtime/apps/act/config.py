"""Git-tracked ACT joint-state runtime configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    load_toml,
    referenced_config,
    repo_path as _repo_path,
    required_table as _required_table,
    string as _string,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.apps.act.config_runtime import bash_config, bridge_argv
from galaxea_a1_runtime.apps.act.config_schema import (
    ActConfig,
    ActExecutionConfig,
    ActPolicyConfig,
    ActRuntimeConfig,
    ActSessionConfig,
)

DEFAULT_ACT_CONFIG = Path("configs/deployments/act_joint.toml")

__all__ = ["bash_config", "bridge_argv", "load_act_config"]


def default_config_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ACT_CONFIG


def load_act_config(path: Path, *, repo_root: Path | None = None) -> ActConfig:
    path, repo_root, data = load_toml(path, repo_root=repo_root)
    system = load_system_config(referenced_config(data, repo_root), repo_root=repo_root)

    session = _required_table(data, "session")
    runtime = _required_table(data, "runtime")
    policy = _required_table(data, "policy")
    execution = _required_table(data, "execution")
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
            disable_backbone_download=bool(
                policy.get("disable_backbone_download", True)
            ),
            deployment_ready=bool(policy.get("deployment_ready", False)),
        ),
        execution=ActExecutionConfig(
            execute=bool(execution.get("execute", False)),
            step_mode=bool(execution.get("step_mode", True)),
            execute_steps_per_inference=int(
                execution.get("execute_steps_per_inference", 8)
            ),
            control_hz=float(execution.get("control_hz", 30.0)),
            max_model_calls=int(execution.get("max_model_calls", 0)),
            print_actions=bool(execution.get("print_actions", True)),
            preview_steps=int(execution.get("preview_steps", 5)),
        ),
    )
    validate_act_config(config)
    return config


def validate_act_config(config: ActConfig) -> None:
    if config.execution.execute_steps_per_inference <= 0:
        raise ValueError("execution.execute_steps_per_inference must be positive")
    if config.execution.control_hz <= 0:
        raise ValueError("execution.control_hz must be positive")
    if config.execution.max_model_calls < 0:
        raise ValueError("execution.max_model_calls must be >= 0")
    if config.execution.preview_steps <= 0:
        raise ValueError("execution.preview_steps must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read A1 ACT joint inference TOML config."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Emit bash assignments for a1_act_joint_runtime.sh",
    )
    args = parser.parse_args(argv)

    config = load_act_config(args.config, repo_root=args.repo_root)
    if args.shell:
        print(bash_config(config))
    else:
        print(config.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
