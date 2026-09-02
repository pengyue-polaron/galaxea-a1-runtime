"""Shared lifecycle for terminal, Web, and Foxglove workflow surfaces."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from embodied_ops.operator_panel import (
    OperatorPanelApplication,
    serve_operator_panel_application,
)

from galaxea_a1_runtime.console import info, success

from .adapter import A1OperatorPanelAdapter
from galaxea_a1_runtime.runtime.operator_session import (
    OperatorSessionClient,
    OperatorSessionServer,
    OperatorSessionUnavailable,
)


def serve_a1_operator_panel(repo_root: Path) -> int:
    """Serve the Web panel and the private native-control session together."""

    adapter = A1OperatorPanelAdapter(repo_root)
    application = OperatorPanelApplication(adapter)
    session = OperatorSessionServer(application)
    session.start()
    try:
        return serve_operator_panel_application(
            application,
            bind=adapter.panel_bind,
            port=adapter.panel_port,
        )
    finally:
        session.close()


def run_collection_session(
    repo_root: Path,
    *,
    config: Path,
    experiment: str,
    task: str,
) -> int:
    """Start collection through the one shared session and follow its logs."""

    root = repo_root.resolve()
    adapter = A1OperatorPanelAdapter(root)
    values = {
        "config": _catalog_config_reference(root, config),
        "experiment": experiment,
        "task": task,
    }
    client = OperatorSessionClient(timeout_s=0.5)
    try:
        client.status()
    except OperatorSessionUnavailable:
        return _run_owned_collection(adapter, values)

    status = client.start("collect", values)
    success("Collection submitted to the active Operator Session.")
    return _follow_workflow(
        client.status,
        lambda run_id: client.stop(run_id=run_id),
        status,
    )


def _run_owned_collection(
    adapter: A1OperatorPanelAdapter,
    values: dict[str, str],
) -> int:
    application = OperatorPanelApplication(adapter)
    session = OperatorSessionServer(application)
    try:
        session.start()
        status = application.start({"workflow": "collect", "values": values})
        success("Collection Operator Session ready for Foxglove control.")
        return _follow_workflow(
            application.workflow.snapshot,
            lambda run_id: application.workflow.stop(run_id=run_id),
            status,
        )
    finally:
        errors: list[str] = []
        status = application.workflow.snapshot()
        if status["active"]:
            try:
                application.workflow.stop(run_id=status["run_id"])
            except RuntimeError as exc:
                errors.append(f"workflow stop failed: {exc}")
        try:
            session.close()
        except RuntimeError as exc:
            errors.append(f"Operator Session close failed: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


def _follow_workflow(
    status_reader: Callable[[], dict],
    stop: Callable[[str], dict],
    initial: dict,
) -> int:
    run_id = initial["run_id"]
    previous_logs: list[str] = []
    info("Use Galaxea A1 Collection Console in Foxglove for guarded controls.")
    try:
        while True:
            status = status_reader()
            if status.get("run_id") != run_id:
                raise RuntimeError("Operator Session switched to another workflow run")
            logs = status.get("logs")
            if isinstance(logs, list) and all(isinstance(line, str) for line in logs):
                _print_appended_logs(previous_logs, logs)
                previous_logs = list(logs)
            if not status.get("active"):
                exit_code = status.get("exit_code")
                return exit_code if isinstance(exit_code, int) else 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
        info("Interrupting the active collection through its Operator Session.")
        status = stop(run_id)
        exit_code = status.get("exit_code")
        return exit_code if isinstance(exit_code, int) else 130


def _print_appended_logs(previous: list[str], current: list[str]) -> None:
    overlap = min(len(previous), len(current))
    while overlap and previous[-overlap:] != current[:overlap]:
        overlap -= 1
    for line in current[overlap:]:
        print(line, flush=True)


def _catalog_config_reference(root: Path, config: Path) -> str:
    candidate = config if config.is_absolute() else root / config
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
