"""Configuration-driven LingBot end-effector bridge entrypoint."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Protocol

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser, info, warning


REPO_ROOT = Path(__file__).resolve().parents[3]


class _LiveStatus(Protocol):
    def break_line(self) -> None: ...


class _BridgeLifecycle(Protocol):
    live_status: _LiveStatus

    def run(self) -> None: ...

    def close(self) -> None: ...


def parse_args() -> Namespace:
    parser = ArgumentParser(description="LingBot-VA EE-pose bridge for Galaxea A1")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(REPO_ROOT),
        help="Tracked LingBot deployment TOML",
    )
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lingbot_config(args.config, repo_root=REPO_ROOT)
    task = config.task_catalog.task(args.task_id)
    from galaxea_a1_runtime.apps.lingbot.bridge import A1LingBotEEBridge

    info(f"LingBot config: {config.path}")
    info(
        f"LingBot task: {task.task_id} distribution={task.distribution} "
        f"prompt={task.prompt!r}"
    )
    if not config.execution.execute:
        warning("Dry run: execution.execute=false in the deployment config.")
    return run_bridge(A1LingBotEEBridge(config, task))


def run_bridge(bridge: _BridgeLifecycle) -> int:
    """Run one bridge and treat an operator interrupt as a clean shutdown."""

    try:
        bridge.run()
        return 0
    except KeyboardInterrupt:
        bridge.live_status.break_line()
        info("Ctrl+C received; locking the arm and finalizing AgentView video.")
        return 0
    finally:
        bridge.close()
