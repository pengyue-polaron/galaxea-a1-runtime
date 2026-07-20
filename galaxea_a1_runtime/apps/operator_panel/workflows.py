"""Build validated argv-only launches for A1 panel workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operator_panel.contracts import InputAction, WorkflowLaunch

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.operator_input import validate_scene_note
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.apps.teleop.collection_task import normalize_collection_task
from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.teleop.config import load_teleop_config


def build_a1_workflow_launch(
    repo_root: Path, workflow: str, values: dict[str, Any]
) -> WorkflowLaunch:
    root = repo_root.resolve()
    if workflow == "collect":
        config_path = _repository_config(root, values.get("config"), "configs/teleop")
        config = load_teleop_config(config_path, repo_root=root)
        experiment = validate_experiment_name(_text(values, "experiment"))
        task = normalize_collection_task(_text(values, "task"))
        return WorkflowLaunch(
            workflow="collect",
            name=f"collect:{experiment}",
            command=(
                str(root / "scripts/apps/teleop/a1_teleop_runtime.sh"),
                "--config",
                str(config.path),
                "--task",
                task,
                "collect",
                experiment,
            ),
            input_actions=(
                InputAction("enter", "Next / Save", "\n", "primary"),
                InputAction("discard", "Discard", "d\n", "danger"),
                InputAction("quit", "Quit", "q\n", "quiet"),
            ),
        )

    if workflow == "evaluate":
        config_path = _repository_config(
            root, values.get("config"), "configs/deployments/lingbot"
        )
        model = _optional_text(values.get("model"))
        config = load_lingbot_config(
            config_path,
            repo_root=root,
            model_selector=model,
        )
        task_id = _text(values, "task")
        config.task_catalog.task(task_id)
        scene_note = validate_scene_note(_text(values, "scene_note"))
        command = [
            str(root / "scripts/apps/lingbot/a1_lingbot_runtime.sh"),
            "--config",
            str(config.path),
        ]
        if model is not None:
            command.extend(("--model", model))
        command.extend(("--task", task_id, "--scene-note", scene_note, "run"))
        return WorkflowLaunch(
            workflow="evaluate",
            name=f"evaluate:{task_id}",
            command=tuple(command),
            input_actions=(
                InputAction("enter", "Next", "\n", "primary"),
                InputAction("quit", "Quit", "q\n", "quiet"),
            ),
        )

    if workflow == "batch":
        plan_path = _repository_config(
            root, values.get("config"), "configs/runs/lingbot"
        )
        model = _optional_text(values.get("model"))
        config = load_lingbot_batch_config(
            plan_path,
            repo_root=root,
            model_selector=model,
        )
        scene_note = validate_scene_note(_text(values, "scene_note"))
        resume = values.get("resume", False)
        if not isinstance(resume, bool):
            raise ValueError("resume must be a boolean")
        command = [
            str(root / "scripts/apps/lingbot/a1_lingbot_runtime.sh"),
            "--config",
            str(config.deployment_path),
        ]
        if model is not None:
            command.extend(("--model", model))
        command.extend(("--scene-note", scene_note, "batch"))
        if resume:
            command.append("--resume")
        command.append(str(config.path))
        return WorkflowLaunch(
            workflow="batch",
            name=f"batch:{config.batch_id}",
            command=tuple(command),
            input_actions=(
                InputAction("enter", "Next / Count", "\n", "primary"),
                InputAction("discard", "Discard", "d\n", "danger"),
                InputAction("quit", "Quit", "q\n", "quiet"),
            ),
        )

    if workflow == "reset":
        system = load_system_config(root / SYSTEM_CONFIG, repo_root=root)
        pose_path = _repository_config(root, values.get("pose"), "configs/poses")
        pose = load_a1_home_pose(pose_path, system=system, repo_root=root)
        return WorkflowLaunch(
            workflow="reset",
            name=f"reset:{pose.path.stem}",
            command=(
                str(root / ".venv/bin/python"),
                str(root / "scripts/runtime/a1_reset.py"),
                "--system-config",
                str(system.path),
                "--pose",
                str(pose.path),
            ),
        )

    if workflow == "camera":
        system = load_system_config(root / SYSTEM_CONFIG, repo_root=root)
        action = _text(values, "action")
        if action not in {"start", "stop"}:
            raise ValueError("camera action must be start or stop")
        command = [
            str(root / "scripts/apps/cameras/a1_camera_web_runtime.sh"),
            "--config",
            str(system.path),
        ]
        if action == "stop":
            command.append("stop")
        return WorkflowLaunch(
            workflow="camera",
            name=f"camera:{action}",
            command=tuple(command),
        )

    raise ValueError(f"unknown operator workflow: {workflow!r}")


def _repository_config(root: Path, value: Any, directory: str) -> Path:
    text = _text_value(value, label="config path")
    path = Path(text)
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    allowed = (root / directory).resolve()
    if not resolved.is_relative_to(allowed) or resolved.suffix != ".toml":
        raise ValueError(f"config must be a repository TOML under {directory}: {text}")
    if not resolved.is_file():
        raise FileNotFoundError(f"repository config is missing: {resolved}")
    return resolved


def _text(values: dict[str, Any], key: str) -> str:
    return _text_value(values.get(key), label=key)


def _text_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return _text_value(value, label="model")
