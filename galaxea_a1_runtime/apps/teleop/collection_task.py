"""Pure preparation of one collection experiment's task identity."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.filesystem import atomic_write_text


def normalize_collection_task(value: str) -> str:
    task = value.strip()
    if not task:
        raise ValueError("collection task must not be empty")
    if "\n" in task or "\r" in task:
        raise ValueError("collection task must be a single line")
    return task


def read_collection_task(experiment_dir: Path) -> str | None:
    task_path = experiment_dir / "task.txt"
    if not task_path.is_file():
        return None
    task = task_path.read_text().strip()
    return normalize_collection_task(task) if task else None


def prepare_collection_task(experiment_dir: Path, value: str) -> str:
    """Create the task identity once and reject accidental task drift."""

    task = normalize_collection_task(value)
    existing = read_collection_task(experiment_dir)
    if existing is not None:
        if existing != task:
            raise ValueError(
                f"collection task mismatch for {experiment_dir.name}: "
                f"existing={existing!r}, requested={task!r}"
            )
        return existing
    experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(experiment_dir / "task.txt", task + "\n")
    return task


def main(argv: list[str] | None = None) -> int:
    from galaxea_a1_runtime.teleop.config import load_teleop_config

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = load_teleop_config(args.config, repo_root=root)
    experiment = validate_experiment_name(args.experiment)
    task = prepare_collection_task(config.collection.data_root / experiment, args.task)
    print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
