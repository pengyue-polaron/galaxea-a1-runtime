"""Choose one tracked deployment prompt before a live policy runtime starts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from galaxea_a1_runtime.configuration.tasks import (
    TaskCatalog,
    TaskPrompt,
    load_task_catalog,
)
from galaxea_a1_runtime.console import ArgumentParser


class TaskSelectionCancelled(RuntimeError):
    pass


def select_task(
    catalog: TaskCatalog,
    *,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> TaskPrompt:
    print(f"[STEP] Select task from {catalog.catalog_id}:", file=output)
    for index, task in enumerate(catalog.tasks, start=1):
        label = task.distribution.upper()
        print(f"  {index}. {task.prompt} [{task.task_id}] [{label}]", file=output)
    print("  q. quit without starting model or hardware", file=output)

    while True:
        output.write("Task > ")
        output.flush()
        try:
            value = input_fn().strip()
        except EOFError as exc:
            raise TaskSelectionCancelled("task selection received EOF") from exc
        if value.lower() in {"q", "quit", "exit"}:
            raise TaskSelectionCancelled("task selection cancelled")
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(catalog.tasks):
                return catalog.tasks[index - 1]
        for task in catalog.tasks:
            if value in {task.task_id, task.prompt}:
                return task
        print(
            "[FAIL] Unknown task. Enter its number, tracked id, exact prompt, or q.",
            file=output,
        )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = load_task_catalog(args.catalog)
    try:
        task = select_task(catalog)
    except TaskSelectionCancelled as exc:
        print(f"[INFO] {exc}.", file=sys.stderr)
        return 2
    print(task.task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
