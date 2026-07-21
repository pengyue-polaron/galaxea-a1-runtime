"""Export an episode-based LeRobot v2.1 dataset from a v3.0 dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embodied_ops.artifacts import atomic_output_directory
from galaxea_a1_runtime.lerobot.dataset_package import (
    dataset_digest,
    json_value,
    portable_metadata_id,
    read_json,
    write_json,
    write_jsonl,
    write_tar_archive,
)

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
CHUNK_SIZE = 1000


def export_v21_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    source_dataset: str,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    final_target_root = target_root.expanduser().resolve()
    with atomic_output_directory(
        final_target_root, overwrite=overwrite
    ) as staging_root:
        return _build_v21_dataset(
            source_root=source_root,
            target_root=staging_root,
            final_target_root=final_target_root,
            repo_id=repo_id,
            source_dataset=source_dataset,
            archive_path=archive_path,
        )


def _build_v21_dataset(
    *,
    source_root: Path,
    target_root: Path,
    final_target_root: Path,
    repo_id: str,
    source_dataset: str,
    archive_path: Path | None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    source_dataset = portable_metadata_id(source_dataset, label="source dataset")
    info = read_json(source_root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("v2.1 export source must be a LeRobot v3.0 dataset")
    (target_root / "meta").mkdir(parents=True)

    frames = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(source_root.glob("data/**/*.parquet"))
        ],
        ignore_index=True,
    )
    episode_meta = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(source_root.glob("meta/episodes/**/*.parquet"))
        ],
        ignore_index=True,
    ).sort_values("episode_index")
    task_frame = pd.read_parquet(source_root / "meta/tasks.parquet")
    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]

    tasks = _task_records(task_frame)
    write_jsonl(target_root / "meta/tasks.jsonl", tasks)
    task_by_index = {record["task_index"]: record["task"] for record in tasks}
    metadata_by_episode = {
        int(row["episode_index"]): row for _, row in episode_meta.iterrows()
    }
    episode_records, episode_stats_records = _write_episode_data(
        frames=frames,
        episode_meta=episode_meta,
        task_by_index=task_by_index,
        target_root=target_root,
    )

    write_jsonl(target_root / "meta/episodes.jsonl", episode_records)
    write_jsonl(target_root / "meta/episodes_stats.jsonl", episode_stats_records)

    video_count = _write_episode_videos(
        source_root=source_root,
        target_root=target_root,
        info=info,
        video_keys=video_keys,
        episode_records=episode_records,
        metadata_by_episode=metadata_by_episode,
    )

    v21_info = _v21_info(info, video_keys=video_keys)
    write_json(target_root / "meta/info.json", v21_info)
    shutil.copy2(source_root / "meta/stats.json", target_root / "meta/stats.json")
    for filename in ("TRAINING.md",):
        source_file = source_root / filename
        if source_file.is_file():
            shutil.copy2(source_file, target_root / filename)
    _copy_source_provenance(source_root, target_root)
    source_manifest = source_root / "meta/eef.json"
    if source_manifest.is_file():
        manifest = read_json(source_manifest)
        source_format = str(manifest.get("format", ""))
        if not source_format.startswith("lerobot_v3_"):
            raise ValueError(
                f"invalid v3 representation manifest format: {source_format!r}"
            )
        intermediate_v3_package_sha256 = manifest.pop("package_sha256", None)
        manifest.pop("archive", None)
        manifest.pop("archive_sha256", None)
        manifest["format"] = source_format.replace("lerobot_v3_", "lerobot_v2.1_", 1)
        manifest["repo_id"] = repo_id
        manifest["source_dataset"] = source_dataset
        manifest["v21_video_codec"] = "h264"
        manifest["conversion_intermediate"] = {
            "format": "lerobot_v3.0",
            "package_sha256": intermediate_v3_package_sha256,
        }
        write_json(target_root / "meta" / source_manifest.name, manifest)

    result = {
        "format": "v2.1",
        "repo_id": repo_id,
        "root": str(final_target_root),
        "episodes": len(episode_records),
        "frames": len(frames),
        "videos": video_count,
        "camera_keys": video_keys,
        "sha256": dataset_digest(target_root),
    }
    if archive_path is not None:
        archive_path, archive_sha256 = write_tar_archive(
            target_root,
            archive_path=archive_path,
            root_name=final_target_root.name,
        )
        result["archive"] = str(archive_path)
        result["archive_sha256"] = archive_sha256
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


def _write_episode_data(
    *,
    frames: pd.DataFrame,
    episode_meta: pd.DataFrame,
    task_by_index: dict[int, str],
    target_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    for _, metadata in episode_meta.iterrows():
        episode_index = int(metadata["episode_index"])
        episode_frames = frames[frames["episode_index"] == episode_index].copy()
        if episode_frames.empty:
            raise ValueError(f"v3 metadata references empty episode {episode_index}")
        episode_frames = episode_frames.sort_values("frame_index")
        expected = np.arange(len(episode_frames), dtype=np.int64)
        if not np.array_equal(episode_frames["frame_index"].to_numpy(), expected):
            raise ValueError(f"episode {episode_index} has non-contiguous frame_index")
        data_path = target_root / V21_DATA_PATH.format(
            episode_chunk=episode_index // CHUNK_SIZE,
            episode_index=episode_index,
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        episode_frames.to_parquet(data_path, index=False)
        task_indices = sorted({int(value) for value in episode_frames["task_index"]})
        records.append(
            {
                "episode_index": episode_index,
                "tasks": [task_by_index[index] for index in task_indices],
                "length": len(episode_frames),
            }
        )
        stats_records.append(
            {
                "episode_index": episode_index,
                "stats": _episode_stats_from_row(metadata),
            }
        )
    return records, stats_records


def _write_episode_videos(
    *,
    source_root: Path,
    target_root: Path,
    info: dict[str, Any],
    video_keys: list[str],
    episode_records: list[dict[str, Any]],
    metadata_by_episode: dict[int, pd.Series],
) -> int:
    count = 0
    fps = int(info["fps"])
    for video_key in video_keys:
        for record in episode_records:
            episode_index = int(record["episode_index"])
            metadata = metadata_by_episode[episode_index]
            source_video = source_root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(metadata[f"videos/{video_key}/chunk_index"]),
                file_index=int(metadata[f"videos/{video_key}/file_index"]),
            )
            start_frame = round(
                float(metadata[f"videos/{video_key}/from_timestamp"]) * fps
            )
            length = int(record["length"])
            target_video = target_root / V21_VIDEO_PATH.format(
                episode_chunk=episode_index // CHUNK_SIZE,
                video_key=video_key,
                episode_index=episode_index,
            )
            target_video.parent.mkdir(parents=True, exist_ok=True)
            _slice_video(
                source_video=source_video,
                target_video=target_video,
                start_frame=start_frame,
                frame_count=length,
                fps=fps,
            )
            actual_frames = _probe_video_frames(target_video)
            if actual_frames != length:
                raise RuntimeError(
                    f"video frame mismatch for {video_key} episode {episode_index}: "
                    f"expected {length}, got {actual_frames}"
                )
            count += 1
    return count


def _v21_info(source_info: dict[str, Any], *, video_keys: list[str]) -> dict[str, Any]:
    features = json.loads(json.dumps(source_info["features"]))
    for key in video_keys:
        height, width, channels = features[key]["shape"]
        features[key]["info"] = {
            "video.height": int(height),
            "video.width": int(width),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(source_info["fps"]),
            "video.channels": int(channels),
            "has_audio": False,
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": source_info.get("robot_type"),
        "total_episodes": int(source_info["total_episodes"]),
        "total_frames": int(source_info["total_frames"]),
        "total_tasks": int(source_info["total_tasks"]),
        "total_videos": int(source_info["total_episodes"]) * len(video_keys),
        "total_chunks": (int(source_info["total_episodes"]) + CHUNK_SIZE - 1)
        // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": int(source_info["fps"]),
        "splits": {"train": f"0:{int(source_info['total_episodes'])}"},
        "data_path": V21_DATA_PATH,
        "video_path": V21_VIDEO_PATH if video_keys else None,
        "features": features,
    }


def _task_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task, row in frame.sort_values("task_index").iterrows():
        records.append({"task_index": int(row["task_index"]), "task": str(task)})
    return records


def _episode_stats_from_row(row: pd.Series) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column, value in row.items():
        if not isinstance(column, str) or not column.startswith("stats/"):
            continue
        _, feature, statistic = column.split("/", 2)
        result.setdefault(feature, {})[statistic] = json_value(value)
    return result


def _slice_video(
    *,
    source_video: Path,
    target_video: Path,
    start_frame: int,
    frame_count: int,
    fps: int,
) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_frame / fps:.9f}",
        "-i",
        str(source_video),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(target_video),
    ]
    subprocess.run(command, check=True)


def _probe_video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())
