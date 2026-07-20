"""Small shared primitives for deterministic LeRobot dataset packages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from galaxea_a1_runtime.filesystem import (
    atomic_output_file,
    atomic_write_text,
)


def portable_metadata_id(value: str, *, label: str) -> str:
    """Validate a logical metadata identifier without host-specific paths."""

    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty logical identifier")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{label} must not be an absolute path: {value!r}")
    return value


def copy_dataset_tree(
    source: Path,
    target: Path,
    *,
    skip_roots: tuple[str, ...] = ("images",),
    hardlink_roots: tuple[str, ...] = ("videos",),
    require_hardlinks: bool = False,
) -> None:
    """Copy a dataset tree, hard-linking large immutable payloads when possible."""
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        if relative.parts[0] in skip_roots:
            continue
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts[0] not in hardlink_roots:
            shutil.copy2(source_path, target_path)
            continue
        try:
            os.link(source_path, target_path)
        except OSError as exc:
            if require_hardlinks:
                raise RuntimeError(
                    "dataset snapshot requires hard-link support for immutable "
                    f"payload {source_path}"
                ) from exc
            shutil.copy2(source_path, target_path)


def namespace_source_provenance(target_root: Path) -> None:
    """Keep direct-recording provenance as source metadata in a derivative."""

    direct = target_root / "meta/galaxea_a1.json"
    namespaced = target_root / "meta/source_galaxea_a1.json"
    if not direct.exists():
        return
    if namespaced.exists():
        raise ValueError("derivative contains ambiguous Galaxea source provenance")
    direct.replace(namespaced)


def vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "count": [len(x)],
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q10": np.quantile(x, 0.10, axis=0).tolist(),
        "q50": np.quantile(x, 0.50, axis=0).tolist(),
        "q90": np.quantile(x, 0.90, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def rewrite_episode_vector_stats(
    target_root: Path,
    *,
    episode_actions: dict[int, np.ndarray],
    episode_states: dict[int, np.ndarray],
) -> None:
    for path in sorted(target_root.glob("meta/episodes/**/*.parquet")):
        episodes = pd.read_parquet(path)
        for row_index, episode_index in enumerate(episodes["episode_index"].to_numpy()):
            for feature, values in (
                ("action", episode_actions[int(episode_index)]),
                ("observation.state", episode_states[int(episode_index)]),
            ):
                for statistic, statistic_values in vector_stats(values).items():
                    episodes.at[row_index, f"stats/{feature}/{statistic}"] = (
                        statistic_values
                    )
        episodes.to_parquet(path, index=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def json_value(value: Any) -> Any:
    """Convert nested NumPy values into JSON-native values."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(root: Path, *, exclude: set[Path] | None = None) -> str:
    excluded = exclude or set()
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative in excluded:
            continue
        digest.update(str(relative).encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def write_tar_archive(
    source_root: Path,
    *,
    archive_path: Path,
    root_name: str,
) -> tuple[Path, str]:
    final_archive = archive_path.expanduser().resolve()
    with (
        atomic_output_file(final_archive) as staging_archive,
        tarfile.open(staging_archive, "w:gz") as archive,
    ):
        archive.add(source_root, arcname=root_name)
    sha256 = file_sha256(final_archive)
    atomic_write_text(
        final_archive.with_suffix(final_archive.suffix + ".sha256"),
        f"{sha256}  {final_archive.name}\n",
        encoding="ascii",
    )
    return final_archive, sha256
