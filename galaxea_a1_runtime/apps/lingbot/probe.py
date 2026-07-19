"""Validate a live LingBot service handshake without running inference."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(
        config_path,
        repo_root=repo_root,
        model_selector=args.model,
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("LingBot config does not belong to --repo-root")

    server = config.server
    client = LingBotClient(
        server.host,
        server.port,
        connect_timeout_s=server.connect_timeout_s,
        close_timeout_s=server.close_timeout_s,
        expected_metadata=server_metadata(config),
    )
    client.close()
    success("LingBot live service contract matches the tracked deployment.")
    return 0
