"""Strict create-only JSON task registries for policy deployments."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from galaxea_a1_runtime.configuration.base import (
    integer,
    lower_identifier,
    require_exact_keys,
    string,
    text,
)


CATALOG_SCHEMA_VERSION = 3
PROMPT_SCHEMA_VERSION = 1
TaskDistribution = Literal["train", "ood"]


@dataclass(frozen=True)
class TaskPrompt:
    task_id: str
    prompt: str
    distribution: TaskDistribution


@dataclass(frozen=True)
class TaskCatalog:
    path: Path
    catalog_id: str
    tasks: tuple[TaskPrompt, ...]

    @property
    def default(self) -> TaskPrompt:
        return self.tasks[0]

    def task(self, task_id: str) -> TaskPrompt:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        allowed = ", ".join(item.task_id for item in self.tasks)
        raise ValueError(f"unknown task id {task_id!r}; expected one of: {allowed}")

    def protocol_contract(self) -> dict[str, Any]:
        return {
            "id": self.catalog_id,
            "tasks": [
                {
                    "id": task.task_id,
                    "prompt": task.prompt,
                    "distribution": task.distribution,
                }
                for task in self.tasks
            ],
        }


def load_task_catalog(path: Path, *, repo_root: Path | None = None) -> TaskCatalog:
    path = _resolve(path, repo_root=repo_root)
    data = _load_json_object(path, label="task catalog")
    require_exact_keys(
        data,
        required={"schema_version", "id"},
        label="task catalog identity",
    )
    if integer(data, "schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"task catalog schema_version must be {CATALOG_SCHEMA_VERSION}"
        )
    catalog_id = lower_identifier(string(data, "id"), label="catalog.id")

    prompt_directory = path.parent / "prompts"
    if not prompt_directory.is_dir():
        raise FileNotFoundError(
            f"task catalog prompt directory is missing: {prompt_directory}"
        )
    visible_entries = sorted(
        entry for entry in prompt_directory.iterdir() if not entry.name.startswith(".")
    )
    invalid_entries = [
        entry.name
        for entry in visible_entries
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json"
    ]
    if invalid_entries:
        raise ValueError(
            f"task catalog prompt directory contains unsupported entries: {invalid_entries}"
        )
    if not visible_entries:
        raise ValueError("task catalog requires at least one prompt JSON file")

    ordered_tasks: list[tuple[int, TaskPrompt]] = []
    for prompt_path in visible_entries:
        order, task = _load_prompt(prompt_path)
        if prompt_path.name != f"{task.task_id}.json":
            raise ValueError(
                f"prompt filename must match its task id: {prompt_path.name!r} != "
                f"{task.task_id}.json"
            )
        ordered_tasks.append((order, task))

    orders = [order for order, _ in ordered_tasks]
    task_ids = [task.task_id for _, task in ordered_tasks]
    prompts = [task.prompt for _, task in ordered_tasks]
    if len(set(orders)) != len(orders):
        raise ValueError("task catalog prompt orders must be unique")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task catalog task ids must be unique")
    if len(set(prompts)) != len(prompts):
        raise ValueError("task catalog prompts must be unique")
    ordered_tasks.sort(key=lambda item: (item[0], item[1].task_id))
    return TaskCatalog(
        path=path,
        catalog_id=catalog_id,
        tasks=tuple(task for _, task in ordered_tasks),
    )


def register_task_prompt(
    catalog_path: Path,
    *,
    task_id: str,
    prompt: str,
    distribution: str,
    repo_root: Path | None = None,
) -> Path:
    """Atomically create one prompt file without modifying existing registry data."""

    catalog_path = _resolve(catalog_path, repo_root=repo_root)
    candidate = _parse_prompt(
        {
            "schema_version": PROMPT_SCHEMA_VERSION,
            "order": 0,
            "id": task_id,
            "prompt": prompt,
            "distribution": distribution,
        },
        label="new prompt",
    )[1]
    with catalog_path.open("rb") as catalog_handle:
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_EX)
        catalog = load_task_catalog(catalog_path, repo_root=repo_root)
        if any(task.task_id == candidate.task_id for task in catalog.tasks):
            raise FileExistsError(
                f"task id is already registered: {candidate.task_id!r}"
            )
        if any(task.prompt == candidate.prompt for task in catalog.tasks):
            raise ValueError(f"prompt is already registered: {candidate.prompt!r}")

        prompt_directory = catalog_path.parent / "prompts"
        current_orders = [
            _load_prompt(path)[0] for path in _prompt_paths(prompt_directory)
        ]
        order = max(current_orders, default=0) + 10
        payload = {
            "schema_version": PROMPT_SCHEMA_VERSION,
            "order": order,
            "id": candidate.task_id,
            "prompt": candidate.prompt,
            "distribution": candidate.distribution,
        }
        target = prompt_directory / f"{candidate.task_id}.json"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"prompt file already exists: {target.name}")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{candidate.task_id}.candidate-",
            suffix=".tmp",
            dir=prompt_directory,
        )
        staging = Path(staging_name)
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            _parse_prompt(
                _load_json_object(staging, label="staged prompt"),
                label="staged prompt",
            )
            try:
                os.link(staging, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"prompt appeared while registering it: {target.name}"
                ) from exc
        finally:
            staging.unlink(missing_ok=True)

        registered = load_task_catalog(catalog_path, repo_root=repo_root).task(
            candidate.task_id
        )
        if registered != candidate:
            raise RuntimeError(
                "registered prompt does not match the validated candidate"
            )
        return target


def _load_prompt(path: Path) -> tuple[int, TaskPrompt]:
    return _parse_prompt(
        _load_json_object(path, label=f"prompt {path.name}"),
        label=f"prompt {path.name}",
    )


def _parse_prompt(data: dict[str, Any], *, label: str) -> tuple[int, TaskPrompt]:
    require_exact_keys(
        data,
        required={"schema_version", "order", "id", "prompt", "distribution"},
        label=label,
    )
    if integer(data, "schema_version") != PROMPT_SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version must be {PROMPT_SCHEMA_VERSION}")
    order = integer(data, "order")
    if order < 0:
        raise ValueError(f"{label} order must be non-negative")
    task_id = lower_identifier(string(data, "id"), label=f"{label}.id")
    prompt = text(data, "prompt")
    if not prompt or prompt != prompt.strip() or "\n" in prompt or "\r" in prompt:
        raise ValueError(
            f"{label}.prompt must be non-empty single-line text without surrounding whitespace"
        )
    distribution = string(data, "distribution")
    if distribution not in {"train", "ood"}:
        raise ValueError(f"{label}.distribution must be 'train' or 'ood'")
    return order, TaskPrompt(
        task_id=task_id,
        prompt=prompt,
        distribution=cast(TaskDistribution, distribution),
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key: {key!r}")
            result[key] = value
        return result

    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


def _prompt_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if not path.name.startswith(".")
            and path.is_file()
            and path.suffix == ".json"
        )
    )


def _resolve(path: Path, *, repo_root: Path | None) -> Path:
    path = path.expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    return path.resolve()
