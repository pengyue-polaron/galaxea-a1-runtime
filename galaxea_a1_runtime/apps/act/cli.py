"""Config-file entrypoint for the ACT joint bridge."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.act.config import default_config_path, load_act_config
from galaxea_a1_runtime.console import ArgumentParser


REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(REPO_ROOT),
        help="Tracked ACT deployment TOML.",
    )
    args = parser.parse_args(argv)
    config = load_act_config(args.config, repo_root=REPO_ROOT)
    from galaxea_a1_runtime.apps.act.bridge import ActJointBridge

    bridge: ActJointBridge | None = None
    try:
        bridge = ActJointBridge(config)
        bridge.run()
        return 0
    finally:
        if bridge is not None:
            bridge.close()
