#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from galaxea_a1_runtime.apps.act.config import load_act_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.configuration.base import load_toml, required_table, string
from galaxea_a1_runtime.configuration.paths import ACT_CONFIG, LINGBOT_CONFIG
from galaxea_a1_runtime.console import ArgumentParser, emit, failure, info, success

MAX_TRACKED_BYTES = 100 * 1024 * 1024


def configured_registry_paths(repo: Path) -> dict[str, Path]:
    """Return config-owned registry paths without resolving their symlink targets."""

    # Typed loading remains the strict validation boundary. Read the same raw
    # path strings afterward because Path.resolve() intentionally follows an
    # already-registered model symlink outside this ignored registry.
    load_lingbot_config(repo / LINGBOT_CONFIG, repo_root=repo)
    load_act_config(repo / ACT_CONFIG, repo_root=repo)
    _, _, lingbot_raw = load_toml(repo / LINGBOT_CONFIG, repo_root=repo)
    _, _, act_raw = load_toml(repo / ACT_CONFIG, repo_root=repo)
    lingbot_policy = required_table(lingbot_raw, "policy_server")
    act_policy = required_table(act_raw, "policy")
    values = {
        "lingbot-base": string(lingbot_policy, "base_model"),
        "lingbot-a1-agentview-square": string(lingbot_policy, "checkpoint"),
        "lingbot-runtime": string(lingbot_policy, "model_root"),
        "act-a1-agentview-square": string(act_policy, "checkpoint"),
    }
    model_root = (repo / "models").resolve()
    configured: dict[str, Path] = {}
    for name, value in values.items():
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo / path
        path = Path(os.path.abspath(path))
        try:
            path.relative_to(model_root)
        except ValueError as exc:
            raise ValueError(
                f"model slot {name} must be configured under models/: {path}"
            ) from exc
        configured[name] = path
    return configured


def configured_slots(repo: Path) -> dict[str, Path]:
    model_root = (repo / "models").resolve()
    return {
        name: path.relative_to(model_root)
        for name, path in configured_registry_paths(repo).items()
        if name != "lingbot-runtime"
    }


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

    def info(self, name: str, detail: str) -> None:
        emit("INFO", f"{name:<24} {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failures += 1
        emit("FAIL", f"{name:<24} {detail}", stream=sys.stderr)


def _require_registry_path(
    reporter: Reporter, repo: Path, name: str, path: Path
) -> Path:
    model_root = (repo / "models").resolve()
    try:
        path.relative_to(model_root)
    except ValueError:
        reporter.fail(name, f"tracked config must use models/: {path}")
    else:
        reporter.pass_(f"{name}_path", str(path.relative_to(repo)))
    return path


def _require_directory(
    reporter: Reporter, name: str, path: Path, required: tuple[str, ...]
) -> None:
    if not path.is_dir():
        reporter.fail(name, f"missing directory: {path}")
        return
    missing = [relative for relative in required if not (path / relative).exists()]
    if missing:
        reporter.fail(name, f"missing: {', '.join(missing)}")
        return
    reporter.pass_(name, f"{path} -> {path.resolve()}")


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


def _check_extra_lingbot_checkpoints(
    reporter: Reporter, repo: Path, configured: Path, expected_size: int
) -> None:
    checkpoint_root = repo / "models/checkpoints/lingbot"
    for checkpoint in sorted(checkpoint_root.glob("*/checkpoint_step_*")):
        if checkpoint == configured:
            continue
        name = f"registered_{checkpoint.parent.name}_{checkpoint.name}"
        _require_directory(
            reporter,
            name,
            checkpoint,
            (
                "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
            ),
        )
        weight = checkpoint / "transformer/diffusion_pytorch_model.safetensors"
        if weight.is_file() and weight.stat().st_size != expected_size:
            reporter.fail(
                f"{name}_size",
                f"expected {expected_size}, got {weight.stat().st_size}",
            )


def doctor(repo: Path) -> int:
    reporter = Reporter()
    lingbot = load_lingbot_config(repo / LINGBOT_CONFIG, repo_root=repo)
    policy = lingbot.policy_server
    configured = configured_registry_paths(repo)

    base = _require_registry_path(
        reporter, repo, "lingbot_base", configured["lingbot-base"]
    )
    checkpoint = _require_registry_path(
        reporter,
        repo,
        "lingbot_checkpoint",
        configured["lingbot-a1-agentview-square"],
    )
    runtime = _require_registry_path(
        reporter, repo, "lingbot_runtime", configured["lingbot-runtime"]
    )
    act_checkpoint = _require_registry_path(
        reporter,
        repo,
        "act_checkpoint",
        configured["act-a1-agentview-square"],
    )

    _require_directory(
        reporter, "lingbot_base", base, ("vae", "text_encoder", "tokenizer")
    )
    _require_directory(
        reporter,
        "lingbot_checkpoint",
        checkpoint,
        ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
    )
    weight = checkpoint / "transformer/diffusion_pytorch_model.safetensors"
    expected_size = policy.expected_weight_size_bytes
    if weight.is_file() and weight.stat().st_size == expected_size:
        reporter.pass_("lingbot_weight_size", _human_bytes(expected_size))
    elif weight.is_file():
        reporter.fail(
            "lingbot_weight_size",
            f"expected {expected_size}, got {weight.stat().st_size}",
        )
    _check_extra_lingbot_checkpoints(reporter, repo, checkpoint, expected_size)

    _require_directory(
        reporter, "act_checkpoint", act_checkpoint, ("config.json", "model.safetensors")
    )
    if runtime.exists():
        reporter.info("lingbot_runtime", f"generated directory present: {runtime}")
    else:
        reporter.info("lingbot_runtime", "generated on first LingBot server start")

    _check_git_storage(reporter, repo)
    if reporter.failures:
        failure(f"Model storage doctor failed with {reporter.failures} error(s).")
        return 1
    success("Model storage is ready.")
    return 0


def register(repo: Path, slot: str, source_arg: str) -> int:
    source = Path(source_arg).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"model source must be a directory: {source}")
    slots = configured_slots(repo)
    if slot not in slots:
        raise ValueError(
            f"unknown model slot {slot!r}; expected one of {sorted(slots)}"
        )
    destination = repo / "models" / slots[slot]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise FileExistsError(f"slot already points elsewhere: {destination}")
        info(f"Model slot already registered: {destination} -> {source}")
        return 0
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing model slot: {destination}")
    destination.symlink_to(source, target_is_directory=True)
    success(f"Registered model slot: {destination} -> {source}")
    return 0


def main() -> int:
    parser = ArgumentParser(description="Manage the ignored local A1 model registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--repo-root", type=Path, required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--repo-root", type=Path, required=True)
    register_parser.add_argument("slot")
    register_parser.add_argument("source")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.command == "doctor":
        return doctor(repo)
    return register(repo, args.slot, args.source)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        failure(str(error))
        sys.exit(2)
