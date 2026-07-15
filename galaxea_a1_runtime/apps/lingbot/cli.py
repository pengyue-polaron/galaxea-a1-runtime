"""Configuration-driven LingBot end-effector bridge entrypoint."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser, info, warning


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> Namespace:
    parser = ArgumentParser(description="LingBot-VA EE-pose bridge for Galaxea A1")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(REPO_ROOT),
        help="Tracked LingBot deployment TOML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lingbot_config(args.config, repo_root=REPO_ROOT)
    from galaxea_a1_runtime.apps.lingbot.bridge import A1LingBotEEBridge

    info(f"LingBot config: {config.path}")
    if not config.execution.execute:
        warning("Dry run: execution.execute=false in the deployment config.")
    bridge = A1LingBotEEBridge(config)
    try:
        bridge.run()
        return 0
    finally:
        bridge.close()
