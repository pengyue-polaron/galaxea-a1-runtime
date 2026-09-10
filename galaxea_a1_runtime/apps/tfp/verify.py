"""Read-only verification for a transferred TFP-Ultra deployment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from galaxea_a1_runtime.apps.tfp.config import default_config_path, load_tfp_config
from galaxea_a1_runtime.apps.tfp.config_schema import TFPConfig
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, success
from galaxea_a1_runtime.models.backend import (
    isolated_backend_pythonpath,
    verify_backend_environment,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployment(config: TFPConfig) -> None:
    if not config.deployment_ready:
        raise RuntimeError("TFP deployment refuses deployment.ready=false")
    verify_backend_environment(config.backend)
    checkpoint = config.model.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"transferred TFP checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size != config.model.checkpoint_size:
        raise ValueError("TFP checkpoint size does not match tracked configuration")
    if file_sha256(checkpoint) != config.model.checkpoint_sha256:
        raise ValueError("TFP checkpoint SHA256 does not match tracked configuration")
    for name, expected in (
        ("info.json", config.model.metadata_info_sha256),
        ("stats.json", config.model.metadata_stats_sha256),
    ):
        path = config.model.metadata / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"TFP metadata mismatch: {path}")
    _verify_environment_imports(config)
    success(
        "TFP-Ultra deployment verified: "
        f"source={config.backend.source.revision} step={config.model.checkpoint_step} "
        f"checkpoint_sha256={config.model.checkpoint_sha256}"
    )


def _verify_environment_imports(config: TFPConfig) -> None:
    expected = dict(config.engine.package_versions)
    code = (
        "import importlib.metadata as m, json, torch; "
        f"names={list(expected)!r}; "
        "versions={name:m.version(name) for name in names}; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "print(json.dumps(versions, sort_keys=True))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": isolated_backend_pythonpath(config.repo_root),
    }
    output = subprocess.run(
        [str(config.backend.environment.python), "-c", code],
        cwd=config.backend.source.checkout,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    actual = json.loads(output)
    if actual != expected:
        raise ValueError(
            f"TFP package version mismatch: expected={expected}, actual={actual}"
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
    verify_deployment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
