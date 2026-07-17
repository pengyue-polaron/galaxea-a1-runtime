"""Set up the pinned OpenPI pi0.5 backend and immutable model artifact."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from galaxea_a1_runtime.apps.pi05.config import default_config_path, load_pi05_config
from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, step, success
from galaxea_a1_runtime.models.backend import (
    ensure_backend_checkout,
    ensure_backend_environment,
)
from galaxea_a1_runtime.models.store import fetch_artifact
from galaxea_a1_runtime.apps.pi05.verify import validate_training_summary


def ensure_environment_imports(config: Pi05Config) -> None:
    step("Validating OpenPI and JAX GPU imports")
    environment = {
        **os.environ,
        "JAX_PLATFORMS": config.engine.jax_platform,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(config.engine.xla_memory_fraction),
    }
    subprocess.run(
        [
            str(config.backend.environment.python),
            "-c",
            (
                "import jax; import openpi; "
                "devices=jax.devices(); "
                "assert devices and devices[0].platform == 'gpu', devices; "
                "print('OpenPI imports OK:', jax.__version__, devices)"
            ),
        ],
        cwd=config.backend.source.checkout,
        env=environment,
        check=True,
    )


def setup(config: Pi05Config) -> None:
    step(f"Ensuring OpenPI source {config.backend.source.revision}")
    ensure_backend_checkout(config.backend)
    step("Synchronizing the pinned OpenPI uv environment")
    ensure_backend_environment(config.backend)
    ensure_environment_imports(config)
    step(
        f"Fetching {config.model.source.repo_id}@{config.model.source.revision} "
        f"({config.model.source.revision_label})"
    )
    result = fetch_artifact(config.model)
    validate_training_summary(config, result.root)
    success(
        "OpenPI pi0.5 inference ready: "
        f"model={config.model.model_id} step={config.model.checkpoint_step} "
        f"files={result.files} bytes={result.bytes} "
        f"manifest={result.manifest_sha256}"
    )


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
    setup(config)
    return 0
