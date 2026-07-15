"""Pure validation for one staged raw teleoperation episode."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_staged_episode(
    episode_dir: Path, *, frame_count: int, depth_enabled: bool
) -> None:
    if frame_count <= 0:
        raise RuntimeError("staged episode must contain at least one frame")
    csv_path = episode_dir / "frames.csv"
    metadata_path = episode_dir / "metadata.json"
    missing = [path.name for path in (csv_path, metadata_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"staged episode is missing required files: {missing}")

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"staged episode metadata is invalid: {exc}") from exc
    metadata_count = metadata.get("frame_count")
    if isinstance(metadata_count, bool) or metadata_count != frame_count:
        raise RuntimeError(
            f"staged episode metadata frame_count={metadata_count!r}, expected {frame_count}"
        )

    camera_dirs = ("cam0", "cam1", *(("cam0_depth",) if depth_enabled else ()))
    if not depth_enabled and (episode_dir / "cam0_depth").exists():
        raise RuntimeError("staged episode has cam0_depth but metadata disables depth")
    for directory in camera_dirs:
        path = episode_dir / directory
        if not path.is_dir():
            raise RuntimeError(
                f"staged episode is missing camera directory: {directory}"
            )
        suffix = ".png" if directory == "cam0_depth" else ".jpg"
        expected = {f"{index:06d}{suffix}" for index in range(frame_count)}
        actual = {item.name for item in path.iterdir()}
        if actual != expected:
            missing_names = sorted(expected - actual)[:3]
            extra_names = sorted(actual - expected)[:3]
            raise RuntimeError(
                f"staged episode {directory} frame set mismatch: "
                f"missing={missing_names}, extra={extra_names}"
            )

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != frame_count:
        raise RuntimeError(
            f"staged episode CSV has {len(rows)} rows, expected {frame_count}"
        )
    for index, row in enumerate(rows):
        if row.get("frame_index") != str(index):
            raise RuntimeError(
                f"staged episode CSV row {index} has "
                f"frame_index={row.get('frame_index')!r}"
            )
        for directory in camera_dirs:
            suffix = ".png" if directory == "cam0_depth" else ".jpg"
            column = f"{directory}_relpath"
            expected_path = f"{directory}/{index:06d}{suffix}"
            if row.get(column) != expected_path:
                raise RuntimeError(
                    f"staged episode CSV row {index} {column}={row.get(column)!r}, "
                    f"expected {expected_path!r}"
                )
