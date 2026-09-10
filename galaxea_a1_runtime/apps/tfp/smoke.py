"""Run the upstream TFP-Ultra synthetic RGB GPU smoke test without robot I/O."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from galaxea_a1_runtime.apps.tfp.config import default_config_path, load_tfp_config
from galaxea_a1_runtime.apps.tfp.verify import verify_deployment
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.models.backend import isolated_backend_pythonpath


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_tfp_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    verify_deployment(config)
    script = (
        config.backend.source.checkout / "scripts/diffusion_ltc/deployment_smoke.py"
    )
    subprocess.run(
        [
            str(config.backend.environment.python),
            str(script),
            "--checkpoint",
            str(config.model.checkpoint),
            "--metadata",
            str(config.model.metadata),
        ],
        cwd=config.backend.source.checkout,
        env={**os.environ, "PYTHONPATH": isolated_backend_pythonpath(repo_root)},
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
