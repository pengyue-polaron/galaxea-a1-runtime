"""Convert legacy A1 raw episode folders into LeRobotDataset v3.

This converter is for one-way migration of old joint-space raw episodes. New
data collection should record the richer EEF runtime contract directly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.config import DatasetConfig
from galaxea_a1_runtime.lerobot.dataset import create_lerobot_dataset
from galaxea_a1_runtime.schema import ActionMode, CameraSpec, DatasetContract, validate_frame_keys

NON_JOINT_COLS = {
    "frame_index",
    "wall_time_ns",
    "ros_stamp_s",
    "cam0_relpath",
    "cam1_relpath",
    "cam0_depth_relpath",
}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    frame_count: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_specs: tuple[CameraSpec, ...]
    fps: int
    schema_version: str = "legacy_joint_raw"

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.state_names


@dataclass(frozen=True)
class RawDatasetSummary:
    source_root: Path
    task: str
    episodes: tuple[RawEpisode, ...]

    @property
    def total_frames(self) -> int:
        return sum(episode.frame_count for episode in self.episodes)


def clean_task_text(text: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return " ".join(text.split())


def list_episode_dirs(source_root: Path, max_episodes: int = 0) -> list[Path]:
    episodes = sorted(path for path in source_root.glob("episode_*") if path.is_dir())
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def resolve_joint_names(df: pd.DataFrame, metadata: dict) -> tuple[str, ...]:
    candidate = metadata.get("joint_names")
    if isinstance(candidate, list) and all(isinstance(item, str) for item in candidate):
        available = tuple(name for name in candidate if name in df.columns)
        if available:
            return available
    return tuple(column for column in df.columns if column not in NON_JOINT_COLS)


def resolve_state_action_names(df: pd.DataFrame, metadata: dict) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    schema_version = str(metadata.get("schema_version", "legacy_joint_raw"))
    if schema_version == TELEOP_RAW_SCHEMA_VERSION:
        state_names = _metadata_names(metadata, "state_names")
        action_names = _metadata_names(metadata, "action_names")
        _require_columns(df, [f"state.{name}" for name in state_names])
        _require_columns(df, [f"action.{name}" for name in action_names])
        return state_names, action_names, schema_version
    joint_names = resolve_joint_names(df, metadata)
    return joint_names, joint_names, schema_version


def infer_image_shape(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return (height, width, 3)


def discover_raw_dataset(
    *,
    source_root: Path,
    max_episodes: int = 0,
    disable_wrist: bool = False,
    fallback_fps: int = 30,
) -> RawDatasetSummary:
    source_root = source_root.expanduser().resolve()
    task_path = source_root / "task.txt"
    if not task_path.is_file():
        raise FileNotFoundError(f"missing task file: {task_path}")
    task = clean_task_text(task_path.read_text())
    if not task:
        raise ValueError(f"empty task file: {task_path}")

    episodes: list[RawEpisode] = []
    for episode_dir in list_episode_dirs(source_root, max_episodes=max_episodes):
        csv_path = episode_dir / "frames.csv"
        metadata_path = episode_dir / "metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        state_names, action_names, schema_version = resolve_state_action_names(df, metadata)
        if not state_names or not action_names:
            raise ValueError(f"no joint columns found in {csv_path}")
        camera_specs = infer_camera_specs(episode_dir, df, disable_wrist=disable_wrist)
        episodes.append(
            RawEpisode(
                path=episode_dir,
                frame_count=len(df),
                state_names=state_names,
                action_names=action_names,
                camera_specs=camera_specs,
                fps=int(metadata.get("fps_target", fallback_fps)),
                schema_version=schema_version,
            )
        )
    if not episodes:
        raise RuntimeError(f"no usable episode_* folders under {source_root}")
    return RawDatasetSummary(source_root=source_root, task=task, episodes=tuple(episodes))


def infer_camera_specs(
    episode_dir: Path,
    df: pd.DataFrame,
    *,
    disable_wrist: bool,
) -> tuple[CameraSpec, ...]:
    frame_index = int(df["frame_index"].iloc[0]) if "frame_index" in df.columns else 0
    row0 = df.iloc[0]
    cameras = [("front", episode_dir / "cam0" / f"{frame_index:06d}.jpg", False)]
    if not disable_wrist:
        cameras.append(("wrist", episode_dir / "cam1" / f"{frame_index:06d}.jpg", False))
    depth_relpath = row0.get("cam0_depth_relpath")
    if isinstance(depth_relpath, str) and depth_relpath:
        cameras.append(("front_depth", episode_dir / depth_relpath, True))
    specs: list[CameraSpec] = []
    for name, path, is_depth in cameras:
        if not path.is_file():
            raise FileNotFoundError(f"missing camera frame: {path}")
        if is_depth:
            with Image.open(path) as image:
                width, height = image.size
            specs.append(
                CameraSpec(
                    name=name,
                    height=height,
                    width=width,
                    channels=1,
                    is_depth_map=True,
                    depth_unit="mm",
                )
            )
        else:
            height, width, channels = infer_image_shape(path)
            specs.append(CameraSpec(name=name, height=height, width=width, channels=channels))
    return tuple(specs)


def raw_episode_contract(
    *,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    camera_specs: tuple[CameraSpec, ...],
    action_mode: ActionMode = ActionMode.JOINT_ABSOLUTE,
) -> DatasetContract:
    return DatasetContract(
        dataset_format="v3.0",
        action_mode=action_mode,
        state_names=state_names,
        action_names=action_names,
        camera_specs=camera_specs,
    )


def legacy_joint_contract(
    *,
    joint_names: tuple[str, ...],
    camera_specs: tuple[CameraSpec, ...],
) -> DatasetContract:
    return raw_episode_contract(
        state_names=joint_names,
        action_names=joint_names,
        camera_specs=camera_specs,
    )


def convert_raw_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    overwrite: bool = False,
    max_episodes: int = 0,
    disable_wrist: bool = False,
    fallback_fps: int = 30,
) -> RawDatasetSummary:
    summary = discover_raw_dataset(
        source_root=source_root,
        max_episodes=max_episodes,
        disable_wrist=disable_wrist,
        fallback_fps=fallback_fps,
    )
    first = summary.episodes[0]
    contract = raw_episode_contract(
        state_names=first.state_names,
        action_names=first.action_names,
        camera_specs=first.camera_specs,
    )
    target_root = target_root.expanduser().resolve()
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"target root exists: {target_root}")
        shutil.rmtree(target_root)

    dataset = create_lerobot_dataset(
        config=DatasetConfig(repo_id=repo_id, root=target_root, fps=first.fps),
        contract=contract,
    )
    try:
        for episode in summary.episodes:
            if episode.state_names != first.state_names:
                raise ValueError(f"state names changed in {episode.path}")
            if episode.action_names != first.action_names:
                raise ValueError(f"action names changed in {episode.path}")
            for frame in iter_episode_frames(episode=episode, task=summary.task, contract=contract):
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
    finally:
        stop = getattr(dataset, "stop_image_writer", None)
        if callable(stop):
            stop()
    return summary


def iter_episode_frames(
    *,
    episode: RawEpisode,
    task: str,
    contract: DatasetContract,
) -> Iterable[dict]:
    df = pd.read_csv(episode.path / "frames.csv")
    if episode.schema_version == TELEOP_RAW_SCHEMA_VERSION:
        state = df[[f"state.{name}" for name in episode.state_names]].to_numpy(dtype=np.float32)
        action = df[[f"action.{name}" for name in episode.action_names]].to_numpy(dtype=np.float32)
    else:
        state = df[list(episode.joint_names)].to_numpy(dtype=np.float32)
        action = np.empty_like(state)
        action[:-1] = state[1:]
        action[-1] = state[-1]
    timestamps = episode_timestamps(df, fps=episode.fps)
    frame_indices = (
        df["frame_index"].to_numpy(dtype=np.int64)
        if "frame_index" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    for row_index, frame_index in enumerate(frame_indices):
        frame = {
            "observation.state": tuple(float(v) for v in state[row_index]),
            "action": tuple(float(v) for v in action[row_index]),
            "task": task,
            "timestamp": float(timestamps[row_index]),
        }
        for camera in episode.camera_specs:
            frame[camera.feature_key()] = load_camera_frame(
                episode=episode,
                df_row=df.iloc[row_index],
                camera=camera,
                frame_index=int(frame_index),
            )
        validate_frame_keys(frame, contract=contract)
        yield frame


def load_camera_frame(
    *,
    episode: RawEpisode,
    df_row: pd.Series,
    camera: CameraSpec,
    frame_index: int,
) -> np.ndarray:
    if camera.is_depth_map:
        relpath = df_row.get("cam0_depth_relpath")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError(f"missing cam0_depth_relpath for {episode.path} frame {frame_index}")
        return load_depth_image(episode.path / relpath)
    src_dir = "cam0" if camera.name == "front" else "cam1"
    return load_rgb_image(episode.path / src_dir / f"{frame_index:06d}.jpg")


def _metadata_names(metadata: dict, key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"metadata.{key} must be a list of strings")
    return tuple(value)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"frames.csv missing columns: {missing}")


def episode_timestamps(df: pd.DataFrame, *, fps: int) -> np.ndarray:
    if "ros_stamp_s" in df.columns:
        values = df["ros_stamp_s"].to_numpy(dtype=np.float64)
        return (values - values[0]).astype(np.float32)
    if "wall_time_ns" in df.columns:
        values = df["wall_time_ns"].to_numpy(dtype=np.float64) / 1e9
        return (values - values[0]).astype(np.float32)
    return np.arange(len(df), dtype=np.float32) / float(fps)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_depth_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be single-channel: {path}")
    return depth.astype(np.uint16, copy=False)[..., None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--disable-wrist", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        summary = discover_raw_dataset(
            source_root=args.source_root,
            max_episodes=args.max_episodes,
            disable_wrist=args.disable_wrist,
            fallback_fps=args.fps,
        )
        print(
            f"raw episodes={len(summary.episodes)} frames={summary.total_frames} "
            f"task={summary.task!r}"
        )
        first = summary.episodes[0]
        print(f"schema_version={first.schema_version}")
        print(f"state={list(first.state_names)}")
        print(f"action={list(first.action_names)}")
        print(f"cameras={[camera.name for camera in first.camera_specs]}")
        return 0

    convert_raw_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        disable_wrist=args.disable_wrist,
        fallback_fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
