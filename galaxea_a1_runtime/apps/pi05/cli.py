"""Configuration-driven OpenPI pi0.5 EEF bridge entrypoint."""

from __future__ import annotations

from pathlib import Path

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
    args = parser.parse_args(argv)
    config = load_pi05_config(args.config, repo_root=REPO_ROOT)
    from galaxea_a1_runtime.apps.pi05.bridge import A1Pi05EEBridge

    info(f"OpenPI pi0.5 config: {config.path}")
    if not config.execution.execute:
        warning("Dry run: execution.execute=false in the deployment config.")
    bridge = A1Pi05EEBridge(config)
    try:
        bridge.run()
        return 0
    finally:
        bridge.close()
