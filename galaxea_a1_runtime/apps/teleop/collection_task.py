"""Pure preparation of one collection experiment's task identity."""

from __future__ import annotations

from pathlib import Path
import json

from galaxea_a1_runtime.lerobot.direct_recording import (
    PROVENANCE_PATH,
    normalize_dataset_task,
)


def normalize_collection_task(value: str) -> str:
    """Normalize through the authoritative direct-dataset task contract."""

    return normalize_dataset_task(value)


def read_collection_task(experiment_dir: Path) -> str | None:
    task_path = experiment_dir / PROVENANCE_PATH
    if not task_path.is_file():
        return None
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read collection provenance: {task_path}: {exc}"
        ) from exc
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, str):
        raise ValueError(f"collection provenance has no task: {task_path}")
    return normalize_collection_task(task)


def prepare_collection_task(experiment_dir: Path, value: str) -> str:
    """Validate a task and reject drift from a committed direct dataset."""

    task = normalize_collection_task(value)
    existing = read_collection_task(experiment_dir)
    if existing is not None:
        if existing != task:
            raise ValueError(
                f"collection task mismatch for {experiment_dir.name}: "
                f"existing={existing!r}, requested={task!r}"
            )
        return existing
    return task
