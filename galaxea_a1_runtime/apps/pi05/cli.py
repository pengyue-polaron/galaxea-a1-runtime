"""Configuration-driven OpenPI pi0.5 EEF bridge entrypoint."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.eef_policy_cli import run_eef_policy_bridge
from galaxea_a1_runtime.apps.pi05.config import default_config_path, load_pi05_config
from galaxea_a1_runtime.console import ArgumentParser, info, warning


REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(REPO_ROOT),
        help="Tracked pi0.5 deployment TOML",
    )
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    config = load_pi05_config(args.config, repo_root=REPO_ROOT)
    task = config.task_catalog.task(args.task_id)
    from galaxea_a1_runtime.apps.pi05.bridge import A1Pi05EEBridge

    info(f"OpenPI pi0.5 config: {config.path}")
    info(
        f"OpenPI pi0.5 task: {task.task_id} distribution={task.distribution} "
        f"prompt={task.prompt!r}"
    )
    if not config.execution.execute:
        warning("Dry run: execution.execute=false in the deployment config.")
    bridge = A1Pi05EEBridge(config, task)
    return run_eef_policy_bridge(bridge)
