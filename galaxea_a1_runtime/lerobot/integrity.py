"""Hardware-free integrity checks for a committed LeRobotDataset v3 graph."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.lerobot.dataset import LEROBOT_GENERATED_FEATURES
from galaxea_a1_runtime.lerobot.dataset_package import (
    non_negative_json_int,
    read_json,
)
from galaxea_a1_runtime.schema import ACTION_FEATURE_KEY, STATE_FEATURE_KEY


def validate_lerobot_v3_payloads(
    root: Path,
    *,
    info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    expected_task: str,
) -> None:
    """Validate task/episode metadata and every referenced data/video payload."""

    import pandas as pd
    import pyarrow.parquet as parquet

    stats = read_json(root / "meta/stats.json", label="LeRobot stats")
    if ACTION_FEATURE_KEY not in stats or STATE_FEATURE_KEY not in stats:
        raise ValueError("LeRobot stats are missing canonical vector features")

    tasks_path = root / "meta/tasks.parquet"
    try:
        tasks = pd.read_parquet(tasks_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read LeRobot tasks: {tasks_path}: {exc}") from exc
    if list(tasks.columns) != ["task_index"]:
        raise ValueError("LeRobot tasks must contain exactly the task_index column")
    task_records = {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}
    total_tasks = non_negative_json_int(info, "total_tasks")
    if total_tasks != 1 or task_records != {0: expected_task}:
        raise ValueError(
            "direct dataset must contain exactly its canonical collection task"
        )

    episode_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not episode_paths:
        raise ValueError("direct dataset has no LeRobot episode metadata")
    try:
        episodes = pd.concat(
            [pd.read_parquet(path) for path in episode_paths],
            ignore_index=True,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read LeRobot episode metadata: {exc}") from exc
    required = {
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"LeRobot episode metadata is missing {sorted(missing)}")
    if len(episodes) != total_episodes:
        raise ValueError("LeRobot episode metadata count does not match info.json")

    episodes = episodes.sort_values("episode_index")
    actual_indices = [
        _plain_int(value, label="episode_index") for value in episodes["episode_index"]
    ]
    if actual_indices != list(range(total_episodes)):
        raise ValueError("LeRobot episode indices must be contiguous from zero")

    data_template = _path_template(info, "data_path")
    expected_rows_by_data_file: dict[Path, int] = defaultdict(int)
    next_frame = 0
    video_keys = tuple(
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    video_template = _path_template(info, "video_path") if video_keys else None
    for _, row in episodes.iterrows():
        episode_index = _plain_int(row["episode_index"], label="episode_index")
        length = _positive_int(row["length"], label=f"episode {episode_index} length")
        start = _non_negative_int(
            row["dataset_from_index"], label=f"episode {episode_index} start"
        )
        end = _non_negative_int(
            row["dataset_to_index"], label=f"episode {episode_index} end"
        )
        if start != next_frame or end != start + length:
            raise ValueError(
                f"episode {episode_index} has a non-contiguous dataset frame range"
            )
        next_frame = end
        if tuple(str(value) for value in row["tasks"]) != (expected_task,):
            raise ValueError(
                f"episode {episode_index} task metadata does not match provenance"
            )
        data_path = _format_dataset_path(
            root,
            data_template,
            chunk_index=_non_negative_int(
                row["data/chunk_index"], label="data chunk_index"
            ),
            file_index=_non_negative_int(
                row["data/file_index"], label="data file_index"
            ),
        )
        expected_rows_by_data_file[data_path] += length
        _validate_episode_videos(
            root=root,
            row=row,
            columns=episodes.columns,
            video_keys=video_keys,
            video_template=video_template,
        )

    if next_frame != total_frames:
        raise ValueError("LeRobot episode frame ranges do not match info.json")
    for path, expected_rows in expected_rows_by_data_file.items():
        _validate_data_file(path, expected_rows=expected_rows, parquet=parquet)


def _validate_episode_videos(
    *,
    root: Path,
    row: Any,
    columns: Any,
    video_keys: tuple[str, ...],
    video_template: str | None,
) -> None:
    for key in video_keys:
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        if chunk_column not in columns or file_column not in columns:
            raise ValueError(
                f"LeRobot episode metadata is missing video reference {key!r}"
            )
        assert video_template is not None
        video_path = _format_dataset_path(
            root,
            video_template,
            video_key=key,
            chunk_index=_non_negative_int(
                row[chunk_column], label=f"{key} video chunk_index"
            ),
            file_index=_non_negative_int(
                row[file_column], label=f"{key} video file_index"
            ),
        )
        if not video_path.is_file():
            raise ValueError(f"LeRobot video payload is missing: {video_path}")


def _validate_data_file(path: Path, *, expected_rows: int, parquet: Any) -> None:
    if not path.is_file():
        raise ValueError(f"LeRobot data payload is missing: {path}")
    try:
        parquet_file = parquet.ParquetFile(path)
        metadata = parquet_file.metadata
        columns = set(parquet_file.schema_arrow.names)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read LeRobot data payload {path}: {exc}") from exc
    if metadata.num_rows != expected_rows:
        raise ValueError(
            f"LeRobot data row count mismatch for {path}: "
            f"expected={expected_rows}, actual={metadata.num_rows}"
        )
    required = set(LEROBOT_GENERATED_FEATURES) | {
        STATE_FEATURE_KEY,
        ACTION_FEATURE_KEY,
    }
    missing = required - columns
    if missing:
        raise ValueError(f"LeRobot data payload {path} is missing {sorted(missing)}")


def _path_template(info: dict[str, Any], key: str) -> str:
    value = info.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"LeRobot info.{key} must be a non-empty path template")
    return value


def _format_dataset_path(root: Path, template: str, **values: object) -> Path:
    try:
        relative = Path(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid LeRobot payload path template {template!r}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"LeRobot payload path escapes its dataset: {relative}")
    return root / relative


def _plain_int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, bool) or float(value) != result:
        raise ValueError(f"{label} must be an integer")
    return result


def _non_negative_int(value: Any, *, label: str) -> int:
    result = _plain_int(value, label=label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: Any, *, label: str) -> int:
    result = _plain_int(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
