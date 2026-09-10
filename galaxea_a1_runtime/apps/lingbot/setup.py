"""Reproducible host-side setup for the LingBot inference backend."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, step, success, warning
from galaxea_a1_runtime.models.backend import (
    ensure_backend_checkout,
    ensure_backend_environment,
    isolated_backend_pythonpath,
)
from galaxea_a1_runtime.models.store import fetch_artifact
from galaxea_a1_runtime.apps.lingbot.verify import (
    training_provenance_warning,
    validate_training_summary,
)


def ensure_environment_imports(config: LingBotConfig) -> None:
    policy = config.policy_server
    backend = policy.backend
    attention_import = "import flash_attn; " if backend.adapter == "lingbot_va" else ""
    if backend.adapter == "diffusion2one":
        attention_import = "from galaxea_a1_runtime.apps.diffusion2one.server import prepare_optional_imports; prepare_optional_imports(); "
    step("Validating LingBot GPU environment imports")
    subprocess.run(
        [
            str(backend.environment.python),
            "-c",
            (
                "import torch, diffusers, transformers; "
                + attention_import
                + "import wan_va.wan_va_server; "
                "assert torch.cuda.is_available(); "
                "print('LingBot imports OK:', torch.__version__, "
                "torch.version.cuda, torch.cuda.get_device_name(0))"
            ),
        ],
        cwd=backend.source.checkout,
        env={
            **os.environ,
            "PYTHONPATH": isolated_backend_pythonpath(policy.model.repo_root),
        },
        check=True,
    )


def setup(config: LingBotConfig, *, environment_only: bool = False) -> None:
    backend = config.policy_server.backend
    step(f"Ensuring LingBot source {backend.source.revision}")
    ensure_backend_checkout(backend)
    step("Synchronizing the locked LingBot environment")
    ensure_backend_environment(backend)
    ensure_environment_imports(config)
    if environment_only:
        success(f"{backend.backend_id} environment ready: {backend.environment.python}")
        return
    model = config.policy_server.model
    step(
        f"Fetching {model.source.repo_id}@{model.source.revision} "
        f"({model.source.revision_label})"
    )
    result = fetch_artifact(model)
    if config.policy_server.base_model is not None:
        fetch_artifact(config.policy_server.base_model)
    provenance = validate_training_summary(config, result.root)
    if message := training_provenance_warning(provenance):
        warning(message)
    success(
        "LingBot inference ready: "
        f"model={model.model_id} files={result.files} bytes={result.bytes} "
        f"manifest={result.manifest_sha256}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="install and verify the backend environment without fetching model weights",
    )
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
    setup(config, environment_only=args.environment_only)
    return 0
