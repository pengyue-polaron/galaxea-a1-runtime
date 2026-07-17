"""Strict tracked task catalogs for multi-prompt policy deployments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from galaxea_a1_runtime.configuration.base import (
    integer,
    load_toml,
    require_exact_keys,
    required_table,
    string,
    text,
)


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
    path, _, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"catalog", "tasks"},
        label="task catalog",
    )
    catalog = required_table(data, "catalog")
    require_exact_keys(
        catalog,
        required={"schema_version", "id"},
        label="task catalog identity",
    )
    if integer(catalog, "schema_version") != 2:
        raise ValueError("task catalog schema_version must be 2")
    catalog_id = _safe_id(string(catalog, "id"), label="catalog.id")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task catalog requires at least one [[tasks]] entry")
    tasks = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise ValueError(f"task catalog entry {index} must be a table")
        require_exact_keys(
            raw_task,
            required={"id", "prompt", "distribution"},
            label=f"task catalog entry {index}",
        )
        task_id = _safe_id(string(raw_task, "id"), label=f"tasks[{index}].id")
        prompt = text(raw_task, "prompt")
        if not prompt or prompt != prompt.strip() or "\n" in prompt:
            raise ValueError(
                f"tasks[{index}].prompt must be non-empty single-line text "
                "without surrounding whitespace"
            )
        distribution = string(raw_task, "distribution")
        if distribution not in {"train", "ood"}:
            raise ValueError(f"tasks[{index}].distribution must be 'train' or 'ood'")
        tasks.append(
            TaskPrompt(
                task_id=task_id,
                prompt=prompt,
                distribution=cast(TaskDistribution, distribution),
            )
        )

    task_ids = [task.task_id for task in tasks]
    prompts = [task.prompt for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task catalog task ids must be unique")
    if len(set(prompts)) != len(prompts):
        raise ValueError("task catalog prompts must be unique")
    return TaskCatalog(path=path, catalog_id=catalog_id, tasks=tuple(tasks))


def _safe_id(value: str, *, label: str) -> str:
    if not value or any(
        not (character.islower() or character.isdigit() or character in {"-", "_"})
        for character in value
    ):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")
    return value
