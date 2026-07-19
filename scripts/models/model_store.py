#!/usr/bin/env python3
"""Manage immutable, content-verified model artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from galaxea_a1_runtime.console import ArgumentParser, emit, failure, info, success
from galaxea_a1_runtime.models.config import ModelArtifactConfig, load_model_config
from galaxea_a1_runtime.models.registry import registered_models
from galaxea_a1_runtime.models.store import fetch_artifact, validate_artifact


MAX_TRACKED_BYTES = 100 * 1024 * 1024


def configured_model_configs(repo: Path) -> tuple[ModelArtifactConfig, ...]:
    return registered_models(repo)


def configured_registry_paths(repo: Path) -> dict[str, Path]:
    """Return every config-owned model path without following symlinks."""

    models = configured_model_configs(repo)
    paths = {
        f"model:{model.model_id}@{model.source.revision}": model.artifact_root
        for model in models
    }
    model_root = (repo / "models").resolve()
    for name, path in paths.items():
        try:
            path.relative_to(model_root)
        except ValueError as exc:
            raise ValueError(
                f"model path {name} must be configured under models/: {path}"
            ) from exc
    return paths


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class Reporter:
    def __init__(self) -> None:
        self.failures = 0

    def pass_(self, name: str, detail: str) -> None:
        emit("PASS", f"{name:<24} {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failures += 1
        emit("FAIL", f"{name:<24} {detail}", stream=sys.stderr)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def _check_git_storage(reporter: Reporter, repo: Path) -> None:
    tracked_models = _git_output(repo, "ls-files", "--", "models").splitlines()
    unexpected = [path for path in tracked_models if path != "models/README.md"]
    if unexpected:
        reporter.fail("tracked_model_files", ", ".join(unexpected))
    else:
        reporter.pass_("tracked_model_files", "none")

    oversized: list[str] = []
    for relative in _git_output(repo, "ls-files").splitlines():
        path = repo / relative
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            oversized.append(f"{relative} ({_human_bytes(path.stat().st_size)})")
    if oversized:
        reporter.fail("tracked_files_gt_100m", ", ".join(oversized))
    else:
        reporter.pass_("tracked_files_gt_100m", "none")

    object_stats = {}
    for line in _git_output(repo, "count-objects", "-v").splitlines():
        key, value = line.split(":", maxsplit=1)
        object_stats[key] = value.strip()
    if object_stats.get("garbage") != "0":
        reporter.fail(
            "git_object_garbage",
            f"count={object_stats.get('garbage')} size={object_stats.get('size-garbage')}",
        )
    else:
        reporter.pass_("git_object_garbage", "none")


def doctor(repo: Path) -> int:
    reporter = Reporter()
    for model in configured_model_configs(repo):
        name = model.model_id.replace("/", ":") + "@" + model.source.revision_label
        try:
            result = validate_artifact(model, verify_hashes=True)
        except (FileNotFoundError, ValueError) as exc:
            reporter.fail(name, str(exc))
        else:
            reporter.pass_(
                name,
                f"{result.files} files {_human_bytes(result.bytes)} "
                f"manifest={result.manifest_sha256}",
            )

    _check_git_storage(reporter, repo)
    if reporter.failures:
        failure(f"Model storage doctor failed with {reporter.failures} error(s).")
        return 1
    success("Model storage is ready.")
    return 0


def fetch(repo: Path, model_config: Path) -> int:
    model = load_model_config(model_config, repo_root=repo)
    info(
        f"Fetching {model.model_id} from {model.source.repo_id}@{model.source.revision}"
    )
    result = fetch_artifact(model)
    success(
        f"Model artifact ready: {result.root} "
        f"({_human_bytes(result.bytes)}, {result.files} files)"
    )
    return 0


def verify(repo: Path, model_config: Path) -> int:
    model = load_model_config(model_config, repo_root=repo)
    result = validate_artifact(model, verify_hashes=True)
    success(f"Model artifact verified: {result.root} manifest={result.manifest_sha256}")
    return 0


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--repo-root", type=Path, required=True)
    for command in ("fetch", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--repo-root", type=Path, required=True)
        command_parser.add_argument("model_config", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.command == "doctor":
        return doctor(repo)
    if args.command == "fetch":
        return fetch(repo, args.model_config)
    return verify(repo, args.model_config)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        failure(str(error))
        sys.exit(2)
