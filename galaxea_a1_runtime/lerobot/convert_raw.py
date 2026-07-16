"""Convert current A1 teleop raw episodes into a LeRobot v3 dataset."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from galaxea_a1_runtime.collection.episode_output import validate_staged_episode
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.filesystem import atomic_output_directory
from galaxea_a1_runtime.lerobot.boundary_trim import (
    EpisodeTrimDecision,
    decide_episode_bounds,
    trim_manifest,
)
from galaxea_a1_runtime.lerobot.boundary_trim_config import BoundaryTrimConfig
from galaxea_a1_runtime.lerobot.dataset import DatasetConfig, create_lerobot_dataset
from galaxea_a1_runtime.lerobot.dataset_package import portable_metadata_id, write_json
from galaxea_a1_runtime.schema import (
    ActionMode,
    CameraSpec,
    DatasetContract,
    validate_frame_keys,
)

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


def discover_raw_dataset(*, source_root: Path) -> RawDatasetSummary:
    source_root = source_root.expanduser().resolve()
    task_path = source_root / "task.txt"
    if not task_path.is_file():
        raise FileNotFoundError(f"missing task file: {task_path}")
    task = clean_task_text(task_path.read_text())
    if not task:
        raise ValueError(f"empty task file: {task_path}")

    episode_dirs = sorted(
        path for path in source_root.glob("episode_*") if path.is_dir()
    )
    if not episode_dirs:
        raise RuntimeError(f"no episode_* folders under {source_root}")

    episodes = tuple(
        _load_raw_episode(path, expected_task=task) for path in episode_dirs
    )
    _validate_episode_contracts(episodes)
    return RawDatasetSummary(source_root=source_root, task=task, episodes=episodes)


def _load_raw_episode(episode_dir: Path, *, expected_task: str) -> RawEpisode:
    csv_path = episode_dir / "frames.csv"
    metadata_path = episode_dir / "metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"raw episode is incomplete: {episode_dir} requires frames.csv and metadata.json"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != TELEOP_RAW_SCHEMA_VERSION:
        raise ValueError(
            f"{episode_dir.name} uses {metadata.get('schema_version')!r}; "
            f"only {TELEOP_RAW_SCHEMA_VERSION!r} is supported"
        )
    if clean_task_text(str(metadata.get("task", ""))) != expected_task:
        raise ValueError(f"episode task differs from task.txt: {episode_dir}")
    if metadata.get("action_mode") != ActionMode.JOINT_ABSOLUTE.value:
        raise ValueError(f"episode action_mode must be joint_absolute: {episode_dir}")

    frame_count = _positive_integer(metadata, "frame_count")
    cameras = metadata.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError(f"metadata.cameras must be a list: {metadata_path}")
    depth_enabled = any(
        isinstance(camera, dict) and camera.get("directory") == "cam0_depth"
        for camera in cameras
    )
    validate_staged_episode(
        episode_dir,
        frame_count=frame_count,
        depth_enabled=depth_enabled,
    )

    frame = pd.read_csv(csv_path)
    if len(frame) != frame_count:
        raise ValueError(
            f"{csv_path} has {len(frame)} rows, metadata declares {frame_count}"
        )
    state_names = _metadata_names(metadata, "state_names")
    action_names = _metadata_names(metadata, "action_names")
    _require_columns(frame, [f"state.{name}" for name in state_names])
    _require_columns(frame, [f"action.{name}" for name in action_names])
    return RawEpisode(
        path=episode_dir,
        frame_count=frame_count,
        state_names=state_names,
        action_names=action_names,
        camera_specs=_camera_specs(cameras),
        fps=_positive_integer(metadata, "fps_target"),
    )


def _camera_specs(cameras: list[Any]) -> tuple[CameraSpec, ...]:
    specs: list[CameraSpec] = []
    seen: set[str] = set()
    expected_directories = {
        "front": "cam0",
        "wrist": "cam1",
        "front_depth": "cam0_depth",
    }
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("metadata camera entries must be tables")
        name = camera.get("name")
        if name not in expected_directories or name in seen:
            raise ValueError(f"unsupported or duplicate metadata camera: {name!r}")
        if camera.get("directory") != expected_directories[name]:
            raise ValueError(f"metadata camera {name!r} has an invalid directory")
        seen.add(name)
        is_depth = name == "front_depth"
        specs.append(
            CameraSpec(
                name=name,
                height=_positive_integer(camera, "height"),
                width=_positive_integer(camera, "width"),
                channels=1 if is_depth else 3,
                is_depth_map=is_depth,
                depth_unit="millimeter" if is_depth else None,
            )
        )
    required = {"front", "wrist"}
    if not required.issubset(seen):
        raise ValueError(f"metadata cameras are missing: {sorted(required - seen)}")
    return tuple(specs)


def raw_episode_contract(
    *,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    camera_specs: tuple[CameraSpec, ...],
) -> DatasetContract:
    return DatasetContract(
        dataset_format="v3.0",
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=state_names,
        action_names=action_names,
        camera_specs=camera_specs,
    )


def convert_raw_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    source_dataset: str,
    overwrite: bool = False,
    expected_contract: DatasetContract | None = None,
    trim_config: BoundaryTrimConfig,
) -> RawDatasetSummary:
    return convert_raw_datasets(
        source_roots=(source_root,),
        target_root=target_root,
        repo_id=repo_id,
        source_dataset=source_dataset,
        overwrite=overwrite,
        expected_contract=expected_contract,
        trim_config=trim_config,
    )[0]


def convert_raw_datasets(
    *,
    source_roots: Sequence[Path],
    target_root: Path,
    repo_id: str,
    source_dataset: str,
    overwrite: bool = False,
    expected_contract: DatasetContract | None = None,
    trim_config: BoundaryTrimConfig,
) -> tuple[RawDatasetSummary, ...]:
    """Convert one logical multi-task dataset from one or more Raw v3 roots."""

    source_dataset = portable_metadata_id(source_dataset, label="source dataset")
    resolved_roots = tuple(root.expanduser().resolve() for root in source_roots)
    if not resolved_roots:
        raise ValueError("at least one raw source root is required")
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("raw source roots must not contain duplicates")
    summaries = tuple(
        discover_raw_dataset(source_root=source_root) for source_root in resolved_roots
    )
    episodes = tuple(episode for summary in summaries for episode in summary.episodes)
    _validate_episode_contracts(episodes)
    first = episodes[0]
    contract = raw_episode_contract(
        state_names=first.state_names,
        action_names=first.action_names,
        camera_specs=first.camera_specs,
    )
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(_contract_mismatch(contract, expected_contract))
    trim_decisions = tuple(
        _plan_episode_trims(summary, config=trim_config) for summary in summaries
    )

    target_root = target_root.expanduser().resolve()
    with atomic_output_directory(
        target_root,
        overwrite=overwrite,
        precreate_staging=False,
    ) as staging_root:
        dataset = create_lerobot_dataset(
            config=DatasetConfig(repo_id=repo_id, root=staging_root, fps=first.fps),
            contract=contract,
        )
        try:
            for summary, decisions in zip(summaries, trim_decisions, strict=True):
                for episode, decision in zip(summary.episodes, decisions, strict=True):
                    for frame in iter_episode_frames(
                        episode=episode,
                        task=summary.task,
                        contract=contract,
                        start=decision.start,
                        end=decision.end,
                    ):
                        dataset.add_frame(frame)
                    dataset.save_episode()
            dataset.finalize()
            manifest = trim_manifest(
                decisions=tuple(
                    (f"{summary.source_root.name}/{episode.path.name}", decision)
                    for summary, decisions in zip(
                        summaries, trim_decisions, strict=True
                    )
                    for episode, decision in zip(
                        summary.episodes, decisions, strict=True
                    )
                ),
                fps=first.fps,
                config=trim_config,
            )
            manifest["source_format"] = TELEOP_RAW_SCHEMA_VERSION
            manifest["source_dataset"] = source_dataset
            if len(summaries) == 1:
                manifest["task"] = summaries[0].task
            else:
                manifest["tasks"] = [summary.task for summary in summaries]
            write_json(staging_root / "meta/trim.json", manifest)
        finally:
            stop = getattr(dataset, "stop_image_writer", None)
            if callable(stop):
                stop()
    return summaries


def iter_episode_frames(
    *,
    episode: RawEpisode,
    task: str,
    contract: DatasetContract,
    start: int = 0,
    end: int | None = None,
) -> Iterable[dict[str, Any]]:
    frame = pd.read_csv(episode.path / "frames.csv")
    state = frame[[f"state.{name}" for name in episode.state_names]].to_numpy(
        dtype=np.float32
    )
    action = frame[[f"action.{name}" for name in episode.action_names]].to_numpy(
        dtype=np.float32
    )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError(
            f"episode contains non-finite state/action values: {episode.path}"
        )
    _validate_gripper_values(episode, state=state, action=action)
    stop = len(frame) if end is None else end
    if start < 0 or stop > len(frame) or start >= stop:
        raise ValueError(
            f"invalid source frame bounds [{start}, {stop}) for {episode.path}"
        )

    for row_index in range(start, stop):
        row = frame.iloc[row_index]
        frame_index = int(row["frame_index"])
        output: dict[str, Any] = {
            "observation.state": state[row_index],
            "action": action[row_index],
            "task": task,
        }
        for camera in episode.camera_specs:
            image = _load_camera_frame(episode.path, row, camera)
            expected_shape = (camera.height, camera.width, camera.channels)
            if image.shape != expected_shape:
                raise ValueError(
                    f"{episode.path} frame {frame_index} camera {camera.name} "
                    f"has shape {image.shape}, expected {expected_shape}"
                )
            output[camera.feature_key()] = image
        validate_frame_keys(output, contract=contract)
        yield output


def _plan_episode_trims(
    summary: RawDatasetSummary, *, config: BoundaryTrimConfig
) -> tuple[EpisodeTrimDecision, ...]:
    decisions = []
    for episode in summary.episodes:
        frame = pd.read_csv(episode.path / "frames.csv")
        state = frame[[f"state.{name}" for name in episode.state_names]].to_numpy(
            dtype=np.float64
        )
        action = frame[[f"action.{name}" for name in episode.action_names]].to_numpy(
            dtype=np.float64
        )
        _validate_gripper_values(episode, state=state, action=action)
        decisions.append(
            decide_episode_bounds(
                actions=action,
                states=state,
                action_names=episode.action_names,
                state_names=episode.state_names,
                fps=episode.fps,
                config=config,
            )
        )
    return tuple(decisions)


def _load_camera_frame(
    episode_dir: Path, row: pd.Series, camera: CameraSpec
) -> np.ndarray:
    directory = {
        "front": "cam0",
        "wrist": "cam1",
        "front_depth": "cam0_depth",
    }[camera.name]
    relpath = row.get(f"{directory}_relpath")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"missing {directory}_relpath in {episode_dir}")
    path = episode_dir / relpath
    if camera.is_depth_map:
        with Image.open(path) as image:
            depth = np.asarray(image)
        if depth.ndim != 2:
            raise ValueError(f"depth image must be single-channel: {path}")
        return depth.astype(np.uint16, copy=False)[..., None]
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _validate_episode_contracts(episodes: tuple[RawEpisode, ...]) -> None:
    first = episodes[0]
    for episode in episodes[1:]:
        if episode.state_names != first.state_names:
            raise ValueError(f"state names changed in {episode.path}")
        if episode.action_names != first.action_names:
            raise ValueError(f"action names changed in {episode.path}")
        if episode.camera_specs != first.camera_specs:
            raise ValueError(f"camera contract changed in {episode.path}")
        if episode.fps != first.fps:
            raise ValueError(f"collection FPS changed in {episode.path}")


def _contract_mismatch(actual: DatasetContract, expected: DatasetContract) -> str:
    differences = []
    if actual.state_names != expected.state_names:
        differences.append("state")
    if actual.action_names != expected.action_names:
        differences.append("action")
    if actual.camera_specs != expected.camera_specs:
        differences.append("cameras")
    return (
        "raw dataset does not match the tracked dataset/system contract: "
        + ", ".join(differences or ["unknown difference"])
    )


def _validate_gripper_values(
    episode: RawEpisode, *, state: np.ndarray, action: np.ndarray
) -> None:
    for label, names, values in (
        ("state", episode.state_names, state),
        ("action", episode.action_names, action),
    ):
        try:
            index = names.index("gripper")
        except ValueError as exc:
            raise ValueError(f"raw {label} contract has no gripper") from exc
        gripper = values[:, index]
        if np.any(gripper < -1e-6) or np.any(gripper > 1.0 + 1e-6):
            raise ValueError(
                f"raw episode gripper {label} is outside normalized [0, 1]: "
                f"{episode.path}"
            )


def _metadata_names(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"metadata.{key} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"metadata.{key} contains duplicates")
    return tuple(value)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frames.csv missing columns: {missing}")


def _positive_integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata.{key} must be a positive integer")
    result = int(value)
    if result <= 0 or float(value) != result:
        raise ValueError(f"metadata.{key} must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    from galaxea_a1_runtime.lerobot.pipeline_config import load_pipeline_config

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    pipeline_config = load_pipeline_config(args.config)

    if args.dry_run:
        summaries = tuple(
            discover_raw_dataset(source_root=source_root)
            for source_root in pipeline_config.raw_source_roots
        )
        first = summaries[0].episodes[0]
        print(
            f"raw roots={len(summaries)} "
            f"episodes={sum(len(summary.episodes) for summary in summaries)} "
            f"frames={sum(summary.total_frames for summary in summaries)}"
        )
        print(f"tasks={[summary.task for summary in summaries]}")
        print(f"schema_version={TELEOP_RAW_SCHEMA_VERSION}")
        print(f"state={list(first.state_names)}")
        print(f"action={list(first.action_names)}")
        print(f"cameras={[camera.name for camera in first.camera_specs]}")
        return 0

    convert_raw_datasets(
        source_roots=pipeline_config.raw_source_roots,
        target_root=args.target_root,
        repo_id=args.repo_id,
        source_dataset=pipeline_config.raw_source_id,
        overwrite=args.overwrite,
        expected_contract=pipeline_config.source_contract,
        trim_config=pipeline_config.boundary_trim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
