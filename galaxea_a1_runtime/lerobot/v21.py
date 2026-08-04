"""Galaxea publication around the shared LeRobot v3-to-v2.1 builder."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from embodied_ops.artifacts import atomic_output_directory
from embodied_ops.datasets.lerobot import (
    V21_CHUNK_SIZE as CHUNK_SIZE,
)
from embodied_ops.datasets.lerobot import (
    V21_DATA_PATH,
    V21_VIDEO_PATH,
    build_lerobot_v21_dataset,
    make_lerobot_v21_info,
)

from galaxea_a1_runtime.lerobot.dataset_package import (
    dataset_digest,
    portable_metadata_id,
    read_json,
    write_json,
    write_tar_archive,
)


def export_v21_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    source_dataset: str,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a Galaxea v2.1 derivative."""

    final_target = target_root.expanduser().resolve()
    with atomic_output_directory(final_target, overwrite=overwrite) as staging:
        result = build_lerobot_v21_dataset(source_root, staging)
        _copy_source_provenance(source_root.expanduser().resolve(), staging)
        _rewrite_eef_manifest(
            source_root=source_root.expanduser().resolve(),
            target_root=staging,
            repo_id=repo_id,
            source_dataset=portable_metadata_id(source_dataset, label="source dataset"),
        )
        result.update(
            {
                "repo_id": repo_id,
                "root": str(final_target),
                "sha256": dataset_digest(staging),
            }
        )
        if archive_path is not None:
            archive, sha256 = write_tar_archive(
                staging,
                archive_path=archive_path,
                root_name=final_target.name,
            )
            result["archive"] = str(archive)
            result["archive_sha256"] = sha256
    return result


def _copy_source_provenance(source_root: Path, target_root: Path) -> None:
    candidates = tuple(
        path
        for path in (
            source_root / "meta/galaxea_a1.json",
            source_root / "meta/source_galaxea_a1.json",
        )
        if path.is_file()
    )
    if len(candidates) > 1:
        raise ValueError("v3 source has conflicting Galaxea provenance files")
    if candidates:
        shutil.copy2(candidates[0], target_root / "meta/source_galaxea_a1.json")


def _rewrite_eef_manifest(
    *, source_root: Path, target_root: Path, repo_id: str, source_dataset: str
) -> None:
    source_manifest = source_root / "meta/eef.json"
    if not source_manifest.is_file():
        return
    manifest = read_json(source_manifest)
    source_format = str(manifest.get("format", ""))
    if not source_format.startswith("lerobot_v3_"):
        raise ValueError(
            f"invalid v3 representation manifest format: {source_format!r}"
        )
    intermediate_sha256 = manifest.pop("package_sha256", None)
    manifest.pop("archive", None)
    manifest.pop("archive_sha256", None)
    manifest["format"] = source_format.replace("lerobot_v3_", "lerobot_v2.1_", 1)
    manifest["repo_id"] = repo_id
    manifest["source_dataset"] = source_dataset
    manifest["v21_video_codec"] = "h264"
    manifest["conversion_intermediate"] = {
        "format": "lerobot_v3.0",
        "package_sha256": intermediate_sha256,
    }
    write_json(target_root / "meta/eef.json", manifest)


def _v21_info(source: dict[str, Any], *, video_keys: list[str]) -> dict[str, Any]:
    result = make_lerobot_v21_info(source)
    actual = [
        key
        for key, feature in source["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if actual != video_keys:
        raise ValueError("video keys differ from the source feature contract")
    return result


__all__ = [
    "CHUNK_SIZE",
    "V21_DATA_PATH",
    "V21_VIDEO_PATH",
    "export_v21_dataset",
]
