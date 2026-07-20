"""Validate a live OpenPI pi0.5 server handshake without running inference."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.pi05.client import Pi05Client
from galaxea_a1_runtime.apps.pi05.config import default_config_path, load_pi05_config
from galaxea_a1_runtime.apps.pi05.protocol import server_metadata
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_pi05_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("pi0.5 config does not belong to --repo-root")
    client = Pi05Client(
        config.server.host,
        config.server.port,
        connect_timeout_s=config.server.connect_timeout_s,
        close_timeout_s=config.server.close_timeout_s,
        expected_metadata=server_metadata(config),
    )
    client.close()
    success("OpenPI pi0.5 server contract verified.")
    return 0
