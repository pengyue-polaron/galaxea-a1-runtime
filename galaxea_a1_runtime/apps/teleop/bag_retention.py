"""Dataset-preserving retention for authoritative raw collection bags."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.apps.teleop.bag_contract import MANIFEST_NAME
from galaxea_a1_runtime.console import info, warning


@dataclass(frozen=True)
class RemovedBag:
    path: Path
    bytes_freed: int
    reason: str


def recordings_root(config) -> Path:
    return config.collection.dataset_root.parent / "recordings"


def enforce_raw_retention(config, *, keep: Path | None = None) -> list[RemovedBag]:
    """Delete abandoned bags, then rotate oldest bags under the global cap."""

    policy = config.collection.raw_retention
    if policy is None:
        return []
    root = recordings_root(config).resolve()
    if not root.is_dir():
        return []
    keep_path = keep.resolve() if keep is not None else None

    removed: list[RemovedBag] = []
    retained: list[tuple[Path, dict[str, object]]] = []
    for bag, manifest in _bag_directories(root):
        if _is_abandoned(manifest):
            removed.append(_remove(bag, root=root, reason="abandoned episode"))
        else:
            retained.append((bag, manifest))

    total = _directory_bytes(root)
    if total <= policy.max_bytes:
        return removed
    for bag, manifest in sorted(retained, key=_age_key):
        if total <= policy.max_bytes:
            break
        if manifest.get("status") not in ("finalized", "incomplete"):
            continue
        if keep_path is not None and bag.resolve() == keep_path:
            continue
        item = _remove(bag, root=root, reason="raw retention cap")
        removed.append(item)
        total -= item.bytes_freed
    if total > policy.max_bytes:
        warning(
            f"Raw recordings remain at {total / 1e9:.1f} GB, above the "
            f"{policy.max_bytes / 1e9:.1f} GB cap; remaining bags are "
            "protected, in-flight, or not finalized"
        )
    return removed


def remove_raw_episode(config, episode: Path, *, reason: str) -> RemovedBag | None:
    """Delete one finalized, non-training episode; disabled retention keeps it."""

    if config.collection.raw_retention is None:
        return None
    root = recordings_root(config).resolve()
    path = _check_removable(episode, root)
    manifest = _load_manifest(path)
    if manifest is None or manifest.get("status") != "finalized":
        return None
    return _remove(path, root=root, reason=reason)


def raw_recordings_usage(config) -> tuple[int, int] | None:
    policy = config.collection.raw_retention
    if policy is None:
        return None
    root = recordings_root(config)
    used = _directory_bytes(root) if root.is_dir() else 0
    return used, policy.max_bytes


def announce_removals(removed: list[RemovedBag]) -> None:
    for item in removed:
        info(
            f"Raw retention removed {item.reason}: "
            f"{item.path.parent.name}/{item.path.name} "
            f"({item.bytes_freed / 1e9:.2f} GB)"
        )


def _bag_directories(root: Path) -> list[tuple[Path, dict[str, object]]]:
    found: list[tuple[Path, dict[str, object]]] = []
    for experiment in sorted(root.iterdir()):
        if (
            experiment.name.startswith(".")
            or experiment.is_symlink()
            or not experiment.is_dir()
        ):
            continue
        for bag in sorted(experiment.iterdir()):
            if bag.name.startswith(".") or bag.is_symlink() or not bag.is_dir():
                continue
            manifest = _load_manifest(bag)
            if manifest is not None:
                found.append((bag, manifest))
    return found


def _load_manifest(bag: Path) -> dict[str, object] | None:
    try:
        payload = json.loads((bag / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = ("bag_id", "experiment", "status", "disposition")
    if any(key not in payload for key in expected):
        return None
    if payload.get("experiment") != bag.parent.name:
        return None
    return payload


def _is_abandoned(manifest: dict[str, object]) -> bool:
    return (
        manifest.get("status") == "finalized" and manifest.get("disposition") != "save"
    )


def _age_key(item: tuple[Path, dict[str, object]]) -> tuple[int, str]:
    path, manifest = item
    end_ns = manifest.get("end_ns")
    if isinstance(end_ns, int) and not isinstance(end_ns, bool) and end_ns > 0:
        return (end_ns, path.name)
    try:
        return (path.stat().st_mtime_ns, path.name)
    except OSError:
        return (0, path.name)


def _directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def _check_removable(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or resolved == root or resolved.parent.parent != root:
        raise RuntimeError(
            f"refusing to remove path outside the raw recordings root: {path}"
        )
    return resolved


def _remove(path: Path, *, root: Path, reason: str) -> RemovedBag:
    resolved = _check_removable(path, root)
    size = _directory_bytes(resolved)
    shutil.rmtree(resolved)
    return RemovedBag(resolved, size, reason)
