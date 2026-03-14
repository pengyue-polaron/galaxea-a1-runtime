#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export A1 raw episode folders to LeRobot dataset v2.1 format.

Expected source structure:
  source_root/
    episode_YYYYmmdd_HHMMSS/
      frames.csv
      metadata.json
      cam0/*.jpg
      cam1/*.jpg

Output structure (v2.1):
  output_root/
    data/chunk-000/episode_000000.parquet
    ...
    meta/info.json
    meta/stats.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    images/chunk-000/episode_000000/cam0/*.jpg
    images/chunk-000/episode_000000/cam1/*.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import (
    create_empty_dataset_info,
    get_hf_features_from_features,
    serialize_dict,
    to_parquet_with_hf_images,
    write_info,
    write_stats,
)

V21 = "v2.1"
KNOWN_NON_JOINT_COLUMNS = {"frame_index", "wall_time_ns", "ros_stamp_s", "cam0_relpath", "cam1_relpath"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export A1 raw episodes to LeRobot dataset v2.1 format.")
    parser.add_argument("--source-root", type=Path, required=True, help="Raw A1 episode root directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Export destination directory.")
    parser.add_argument(
        "--image-mode",
        choices=["copy", "symlink", "reference"],
        default="copy",
        help="How image files are stored in output dataset.",
    )
    parser.add_argument("--robot-type", default="a1_single_arm", help="robot_type written to meta/info.json.")
    parser.add_argument("--fps", type=int, default=30, help="Fallback fps when metadata does not provide it.")
    parser.add_argument("--default-task", default="A1 single-arm teleop collection", help="Fallback task name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-root if it already exists.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Export at most N episodes. 0 means all.")
    return parser.parse_args()


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_episode_dirs(source_root: Path, max_episodes: int) -> list[Path]:
    episodes = sorted(p for p in source_root.glob("episode_*") if p.is_dir())
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def resolve_joint_names(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    candidate = metadata.get("joint_names")
    if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
        available = [name for name in candidate if name in df.columns]
        if available:
            return available

    cols = [c for c in df.columns if c not in KNOWN_NON_JOINT_COLUMNS]
    if not cols:
        raise ValueError("No joint columns found in frames.csv.")
    return cols


def infer_image_shape(cam_meta: dict[str, Any], sample_path: Path) -> tuple[int, int, int]:
    height = cam_meta.get("height")
    width = cam_meta.get("width")
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return (height, width, 3)

    with Image.open(sample_path) as img:
        width, height = img.size
    return (height, width, 3)


def prepare_episode_images(
    episode_dir: Path,
    output_root: Path,
    episode_index: int,
    image_mode: str,
) -> tuple[Path, Path]:
    src_cam0 = episode_dir / "cam0"
    src_cam1 = episode_dir / "cam1"
    if not src_cam0.exists() or not src_cam1.exists():
        raise FileNotFoundError(f"Missing cam0/cam1 folders in {episode_dir}.")

    dst_base = output_root / "images" / "chunk-000" / f"episode_{episode_index:06d}"
    dst_cam0 = dst_base / "cam0"
    dst_cam1 = dst_base / "cam1"

    if image_mode == "copy":
        shutil.copytree(src_cam0, dst_cam0, dirs_exist_ok=True)
        shutil.copytree(src_cam1, dst_cam1, dirs_exist_ok=True)
    elif image_mode == "symlink":
        dst_cam0.parent.mkdir(parents=True, exist_ok=True)
        if dst_cam0.exists() or dst_cam0.is_symlink():
            if dst_cam0.is_symlink() or dst_cam0.is_file():
                dst_cam0.unlink()
            else:
                shutil.rmtree(dst_cam0)
        if dst_cam1.exists() or dst_cam1.is_symlink():
            if dst_cam1.is_symlink() or dst_cam1.is_file():
                dst_cam1.unlink()
            else:
                shutil.rmtree(dst_cam1)
        dst_cam0.symlink_to(src_cam0.resolve())
        dst_cam1.symlink_to(src_cam1.resolve())

    return src_cam0, src_cam1


def build_features(
    n_joints: int,
    joint_names: list[str],
    cam0_shape: tuple[int, int, int],
    cam1_shape: tuple[int, int, int],
) -> dict[str, dict[str, Any]]:
    return {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
        "observation.images.cam0": {
            "dtype": "image",
            "shape": cam0_shape,
            "names": ["height", "width", "channels"],
            "info": None,
        },
        "observation.images.cam1": {
            "dtype": "image",
            "shape": cam1_shape,
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }


def main() -> None:
    args = parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output-root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    episode_dirs = list_episode_dirs(source_root, args.max_episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* folders found under {source_root}.")

    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episodes_jsonl: list[dict[str, Any]] = []
    episodes_stats_jsonl: list[dict[str, Any]] = []
    per_episode_stats: list[dict[str, Any]] = []

    features: dict[str, dict[str, Any]] | None = None
    hf_features = None
    total_frames = 0
    global_index_start = 0
    valid_episode_index = 0
    dataset_fps = args.fps

    for episode_dir in tqdm(episode_dirs, desc="export episodes"):
        frames_csv = episode_dir / "frames.csv"
        metadata_json = episode_dir / "metadata.json"
        if not frames_csv.exists() or not metadata_json.exists():
            continue

        with metadata_json.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if valid_episode_index == 0:
            dataset_fps = int(metadata.get("fps_target", args.fps))

        frame_df = pd.read_csv(frames_csv)
        if frame_df.empty:
            continue

        src_cam0, src_cam1 = prepare_episode_images(episode_dir, output_root, valid_episode_index, args.image_mode)

        joint_names = resolve_joint_names(frame_df, metadata)
        obs_state = frame_df[joint_names].to_numpy(dtype=np.float32)
        action = obs_state.copy()

        if "ros_stamp_s" in frame_df.columns:
            timestamp = frame_df["ros_stamp_s"].to_numpy(dtype=np.float32)
        elif "wall_time_ns" in frame_df.columns:
            timestamp = (frame_df["wall_time_ns"].to_numpy(dtype=np.float64) / 1e9).astype(np.float32)
        else:
            timestamp = np.arange(len(frame_df), dtype=np.float32) / float(args.fps)

        if "frame_index" in frame_df.columns:
            frame_index = frame_df["frame_index"].to_numpy(dtype=np.int64)
        else:
            frame_index = np.arange(len(frame_df), dtype=np.int64)

        n_frames = len(frame_df)
        episode_index = np.full(n_frames, valid_episode_index, dtype=np.int64)
        dataset_index = np.arange(global_index_start, global_index_start + n_frames, dtype=np.int64)

        task = str(metadata.get("task", args.default_task))
        task_index = task_to_index.setdefault(task, len(task_to_index))
        task_index_arr = np.full(n_frames, task_index, dtype=np.int64)

        cam0_rel = frame_df["cam0_relpath"].astype(str).tolist()
        cam1_rel = frame_df["cam1_relpath"].astype(str).tolist()
        if args.image_mode == "reference":
            cam0_store_paths = [str((episode_dir / rel).resolve()) for rel in cam0_rel]
            cam1_store_paths = [str((episode_dir / rel).resolve()) for rel in cam1_rel]
            cam0_stats_paths = cam0_store_paths
            cam1_stats_paths = cam1_store_paths
        else:
            cam0_store_paths = [
                str((output_root / "images" / "chunk-000" / f"episode_{valid_episode_index:06d}" / rel).resolve())
                for rel in cam0_rel
            ]
            cam1_store_paths = [
                str((output_root / "images" / "chunk-000" / f"episode_{valid_episode_index:06d}" / rel).resolve())
                for rel in cam1_rel
            ]
            cam0_stats_paths = cam0_store_paths
            cam1_stats_paths = cam1_store_paths

        if features is None:
            cam0_meta = metadata.get("cam0", {}) if isinstance(metadata.get("cam0"), dict) else {}
            cam1_meta = metadata.get("cam1", {}) if isinstance(metadata.get("cam1"), dict) else {}
            cam0_shape = infer_image_shape(cam0_meta, (src_cam0 / Path(cam0_rel[0]).name))
            cam1_shape = infer_image_shape(cam1_meta, (src_cam1 / Path(cam1_rel[0]).name))
            features = build_features(obs_state.shape[1], joint_names, cam0_shape, cam1_shape)
            hf_features = get_hf_features_from_features(features)

        episode_df = pd.DataFrame(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": dataset_index,
                "task_index": task_index_arr,
                "observation.state": obs_state.tolist(),
                "action": action.tolist(),
                "observation.images.cam0": [{"path": p, "bytes": None} for p in cam0_store_paths],
                "observation.images.cam1": [{"path": p, "bytes": None} for p in cam1_store_paths],
            }
        )

        episode_data_path = output_root / "data" / "chunk-000" / f"episode_{valid_episode_index:06d}.parquet"
        to_parquet_with_hf_images(episode_df, episode_data_path, features=hf_features)

        assert features is not None
        episode_stats = compute_episode_stats(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": dataset_index,
                "task_index": task_index_arr,
                "observation.state": obs_state,
                "action": action,
                "observation.images.cam0": cam0_stats_paths,
                "observation.images.cam1": cam1_stats_paths,
            },
            features,
        )
        per_episode_stats.append(episode_stats)
        episodes_stats_jsonl.append({"episode_index": valid_episode_index, "stats": serialize_dict(episode_stats)})
        episodes_jsonl.append({"episode_index": valid_episode_index, "tasks": [task], "length": int(n_frames)})

        total_frames += int(n_frames)
        global_index_start += int(n_frames)
        valid_episode_index += 1

    if valid_episode_index == 0:
        raise RuntimeError("No valid episodes were exported.")

    assert features is not None
    info = create_empty_dataset_info(
        codebase_version=V21,
        fps=dataset_fps,
        features=features,
        use_videos=False,
        robot_type=args.robot_type,
    )
    info["total_episodes"] = valid_episode_index
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_to_index)
    info["total_chunks"] = 1
    info["total_videos"] = 0
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = None
    write_info(info, output_root)

    aggregate = aggregate_stats(per_episode_stats)
    write_stats(aggregate, output_root)

    tasks_jsonl = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_index.items(), key=lambda x: x[1])]
    write_jsonl(tasks_jsonl, output_root / "meta" / "tasks.jsonl")
    write_jsonl(episodes_jsonl, output_root / "meta" / "episodes.jsonl")
    write_jsonl(episodes_stats_jsonl, output_root / "meta" / "episodes_stats.jsonl")

    print(f"Export complete: {output_root}")
    print(f"Episodes: {valid_episode_index}, Frames: {total_frames}, Tasks: {len(task_to_index)}")


if __name__ == "__main__":
    main()
