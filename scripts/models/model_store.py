#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib


SLOTS = {
    "lingbot-base": Path("base/lingbot-va-base"),
    "lingbot-a1-banana-step500": Path(
        "checkpoints/lingbot/a1_banana_in_plate/checkpoint_step_500"
    ),
    "lingbot-a1-banana-step1000": Path(
        "checkpoints/lingbot/a1_banana_in_plate/checkpoint_step_1000"
    ),
    "act-a1-banana-step30000": Path(
        "checkpoints/act/a1_banana_joint_state_30k/checkpoint_step_30000"
    ),
}
MAX_TRACKED_BYTES = 100 * 1024 * 1024


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _configured_path(repo: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.abspath(repo / path))


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
        print(f"[PASS] {name:<24} {detail}")

    def info(self, name: str, detail: str) -> None:
        print(f"[INFO] {name:<24} {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failures += 1
        print(f"[FAIL] {name:<24} {detail}")


def _require_registry_path(
    reporter: Reporter, repo: Path, name: str, raw: str
) -> Path:
    model_root = Path(os.path.abspath(repo / "models"))
    path = _configured_path(repo, raw)
    try:
        path.relative_to(model_root)
    except ValueError:
        reporter.fail(name, f"tracked config must use models/: {raw}")
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
            ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
        )
        weight = checkpoint / "transformer/diffusion_pytorch_model.safetensors"
        if weight.is_file() and weight.stat().st_size != expected_size:
            reporter.fail(
                f"{name}_size",
                f"expected {expected_size}, got {weight.stat().st_size}",
            )


def doctor(repo: Path) -> int:
    reporter = Reporter()
    lingbot = _toml(repo / "configs/inference/lingbot_va_a1.toml")["policy_server"]
    act = _toml(repo / "configs/inference/act_joint_a1.toml")["policy"]

    base = _require_registry_path(reporter, repo, "lingbot_base", lingbot["base_model"])
    checkpoint = _require_registry_path(
        reporter, repo, "lingbot_checkpoint", lingbot["checkpoint"]
    )
    runtime = _require_registry_path(
        reporter, repo, "lingbot_runtime", lingbot["model_root"]
    )
    act_checkpoint = _require_registry_path(
        reporter, repo, "act_checkpoint", act["checkpoint"]
    )

    _require_directory(reporter, "lingbot_base", base, ("vae", "text_encoder", "tokenizer"))
    _require_directory(
        reporter,
        "lingbot_checkpoint",
        checkpoint,
        ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
    )
    weight = checkpoint / "transformer/diffusion_pytorch_model.safetensors"
    expected_size = int(lingbot["expected_weight_size_bytes"])
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
        print(f"Model storage doctor failed with {reporter.failures} error(s).")
        return 1
    print("Model storage is ready.")
    return 0


def register(repo: Path, slot: str, source_arg: str) -> int:
    source = Path(source_arg).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"model source must be a directory: {source}")
    destination = repo / "models" / SLOTS[slot]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise FileExistsError(f"slot already points elsewhere: {destination}")
        print(f"Model slot already registered: {destination} -> {source}")
        return 0
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing model slot: {destination}")
    destination.symlink_to(source, target_is_directory=True)
    print(f"Registered model slot: {destination} -> {source}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the ignored local A1 model registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--repo-root", type=Path, required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--repo-root", type=Path, required=True)
    register_parser.add_argument("slot", choices=tuple(SLOTS))
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
        print(f"[FAIL] {error}", file=sys.stderr)
        sys.exit(2)
