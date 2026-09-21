"""Select one registered LingBot-family target and launch its runtime."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from embodied_ops import TaskPrompt, TaskSelectionCancelled, select_task

from galaxea_a1_runtime.console import ArgumentParser, failure, info, warning
from galaxea_a1_runtime.apps.inference.targets import (
    InferenceTarget,
    discover_targets,
)

ACTIONS = ("run", "server", "smoke")
RUNTIME_SCRIPT = Path("scripts/apps/lingbot/a1_lingbot_runtime.sh")


class _Cancelled(RuntimeError):
    """Interactive selection ended without starting anything."""


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "selector",
        nargs="?",
        help="deployment id, deployment path, or model id; omit to choose one",
    )
    parser.add_argument("--task", help="tracked task id from the target catalog")
    parser.add_argument("--action", choices=ACTIONS, default="run")
    parser.add_argument("--list", action="store_true", help="list targets and exit")
    parser.add_argument("--json", action="store_true", help="print targets as JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved command only"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def run(args: Any) -> int:
    repo_root = args.repo_root.resolve()
    try:
        targets = discover_targets(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 2
    if not targets:
        failure("no LingBot-family deployments are registered")
        return 2
    if args.json:
        print(json.dumps(_targets_json(targets, repo_root), indent=2, sort_keys=True))
        return 0
    if args.list:
        _print_targets(targets)
        return 0
    if args.action != "run" and args.task is not None:
        failure("--task is valid only with --action run")
        return 2
    try:
        target = _select_target(targets, args.selector, repo_root=repo_root)
        task = _resolve_task(target, args.task) if args.action == "run" else None
    except _Cancelled as exc:
        info(str(exc))
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 2
    if target.artifact_state != "ready":
        failure(
            f"artifact is {target.artifact_state}: {target.model.artifact_root}; "
            f"fetch it with: just model-fetch "
            f"{target.model.path.relative_to(repo_root)}"
        )
        return 2
    command = _command(repo_root, target, action=args.action, task=task)
    info(f"Target: {target.deployment_id} ({target.adapter})")
    info(
        f"Model: {target.model.model_id} "
        f"(step {target.model.checkpoint_step}, {target.model.source.revision_label})"
    )
    if task is not None:
        info(f"Task: {task.task_id} [{task.distribution}] — {task.prompt}")
    if args.action == "run":
        if target.execute:
            warning(
                "execution enabled: the run moves the A1 through the relay safety gates"
            )
        else:
            info("execution disabled: camera-input testing only")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    try:
        os.execvpe(str(command[0]), command, os.environ.copy())
    except OSError as exc:
        failure(f"cannot start the runtime: {exc}")
        return 2
    raise AssertionError("unreachable")


def _selectors(target: InferenceTarget, repo_root: Path) -> frozenset[str]:
    model = target.model
    return frozenset(
        {
            target.deployment_id,
            target.path.stem,
            str(target.path.relative_to(repo_root)),
            model.model_id,
            model.model_id.split("/")[-1],
            model.path.stem,
            f"{model.model_id}@{model.source.revision_label}",
        }
    )


def _select_target(
    targets: tuple[InferenceTarget, ...],
    selector: str | None,
    *,
    repo_root: Path,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> InferenceTarget:
    if selector is not None:
        if not selector or selector != selector.strip():
            raise ValueError("selector must be non-empty without surrounding space")
        matches = [
            target for target in targets if selector in _selectors(target, repo_root)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(target.deployment_id for target in matches)
            raise ValueError(f"ambiguous target {selector!r}; choose one of: {choices}")
        available = ", ".join(target.deployment_id for target in targets)
        raise ValueError(f"unknown target {selector!r}; available: {available}")
    _print_targets(targets, output=output)
    while True:
        output.write(f"Select target [1-{len(targets)}] (q to quit) > ")
        output.flush()
        try:
            value = input_fn().strip()
        except EOFError as exc:
            raise _Cancelled("target selection received EOF") from exc
        if value.lower() in {"q", "quit", "exit"}:
            raise _Cancelled("target selection cancelled")
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(targets):
                return targets[index - 1]
        for target in targets:
            if value in _selectors(target, repo_root):
                return target
        print(
            "[FAIL] Unknown target. Enter its number, deployment id, or model id.",
            file=output,
        )


def _resolve_task(
    target: InferenceTarget,
    task_id: str | None,
    *,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> TaskPrompt:
    if task_id is not None:
        return target.catalog.task(task_id)
    if len(target.catalog.tasks) == 1:
        task = target.catalog.tasks[0]
        info(
            f"Catalog {target.catalog.catalog_id} has one prompt; using {task.task_id}"
        )
        return task
    if not sys.stdin.isatty():
        allowed = ", ".join(task.task_id for task in target.catalog.tasks)
        raise ValueError(f"catalog requires --task; expected one of: {allowed}")
    try:
        return select_task(target.catalog, input_fn=input_fn, output=output)
    except TaskSelectionCancelled as exc:
        raise _Cancelled(str(exc)) from exc


def _command(
    repo_root: Path,
    target: InferenceTarget,
    *,
    action: str,
    task: TaskPrompt | None,
) -> list[str]:
    command = [
        str(repo_root / RUNTIME_SCRIPT),
        action,
        "--config",
        str(target.path),
        "--model",
        target.model.model_id,
    ]
    if task is not None:
        command += ["--task", task.task_id]
    return command


def _print_targets(
    targets: tuple[InferenceTarget, ...], *, output: TextIO = sys.stdout
) -> None:
    print("Registered inference targets:", file=output)
    for index, target in enumerate(targets, start=1):
        counts: dict[str, int] = {}
        for task in target.catalog.tasks:
            counts[task.distribution] = counts.get(task.distribution, 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
        flags = "execute" if target.execute else "no-execute"
        print(
            f"  {index:>2}. {target.deployment_id} · "
            f"{target.model.model_id} @{target.model.checkpoint_step}",
            file=output,
        )
        print(
            f"      catalog {target.catalog.catalog_id} ({summary}) · "
            f"artifact {target.artifact_state} · {flags}",
            file=output,
        )


def _targets_json(
    targets: tuple[InferenceTarget, ...], repo_root: Path
) -> list[dict[str, Any]]:
    return [
        {
            "deployment_id": target.deployment_id,
            "deployment": str(target.path.relative_to(repo_root)),
            "adapter": target.adapter,
            "execute": target.execute,
            "ready": target.ready,
            "artifact_state": target.artifact_state,
            "model": {
                "id": target.model.model_id,
                "revision": target.model.source.revision,
                "revision_label": target.model.source.revision_label,
                "checkpoint_step": target.model.checkpoint_step,
                "descriptor": str(target.model.path.relative_to(repo_root)),
            },
            "catalog": {
                "id": target.catalog.catalog_id,
                "path": str(target.catalog.path.relative_to(repo_root)),
                "tasks": [
                    {
                        "id": task.task_id,
                        "prompt": task.prompt,
                        "distribution": task.distribution,
                    }
                    for task in target.catalog.tasks
                ],
            },
        }
        for target in targets
    ]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
