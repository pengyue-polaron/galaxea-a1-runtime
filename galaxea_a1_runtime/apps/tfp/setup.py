"""Set up the pinned TFP-Ultra checkout and isolated inference environment."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

from galaxea_a1_runtime.apps.tfp.config import default_config_path, load_tfp_config
from galaxea_a1_runtime.apps.tfp.verify import verify_deployment
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, step
from galaxea_a1_runtime.models.backend import (
    ensure_backend_checkout,
    ensure_backend_environment,
)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_tfp_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("TFP config does not belong to --repo-root")
    step(f"Ensuring TFP-Ultra source {config.backend.source.revision}")
    ensure_backend_checkout(config.backend)
    step("Synchronizing the pinned TFP-Ultra environment")
    with chdir(repo_root):
        ensure_backend_environment(config.backend)
    verify_deployment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
