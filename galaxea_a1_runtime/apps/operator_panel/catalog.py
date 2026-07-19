"""Build the dynamic A1 form and configuration catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operator_panel.config_store import RepositoryConfigStore

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.configuration.paths import (
    A1_RESET_POSE,
    LINGBOT_BATCH_CONFIG,
    LINGBOT_CONFIG,
    SYSTEM_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.models.registry import registered_models
from galaxea_a1_runtime.teleop.config import load_teleop_config

from .configuration import looks_like_a1_pose


def build_a1_catalog(
    repo_root: Path, config_store: RepositoryConfigStore
) -> dict[str, Any]:
    root = repo_root.resolve()
    system = load_system_config(root / SYSTEM_CONFIG, repo_root=root)
    teleop_options = []
    for path in sorted((root / "configs/teleop").glob("*.toml")):
        config = load_teleop_config(path, repo_root=root)
        teleop_options.append(_option(config.path, root))

    deployment_options = []
    task_options = []
    for path in sorted((root / "configs/deployments/lingbot").glob("*.toml")):
        config = load_lingbot_config(path, repo_root=root)
        reference = _reference(config.path, root)
        deployment_options.append(_option(config.path, root))
        task_options.extend(
            {
                "value": task.task_id,
                "label": f"{task.prompt} · {task.distribution.upper()}",
                "depends_value": reference,
            }
            for task in config.task_catalog.tasks
        )

    batch_options = []
    reset_paths: set[Path] = set()
    for path in sorted((root / "configs/runs/lingbot").glob("*.toml")):
        config = load_lingbot_batch_config(path, repo_root=root)
        reset_paths.add(config.reset_pose)
        batch_options.append(
            {
                "value": _reference(config.path, root),
                "label": f"{config.batch_id} · {config.total_attempts} attempts",
            }
        )

    for path in sorted((root / "configs/poses").glob("*.toml")):
        if looks_like_a1_pose(path):
            load_a1_home_pose(path, system=system, repo_root=root)
            reset_paths.add(path.resolve())
    reset_options = [_option(path, root) for path in sorted(reset_paths)]

    model_options = [{"value": "", "label": "Default from configuration"}]
    model_options.extend(
        {
            "value": model.path.stem,
            "label": f"{model.path.stem} · {model.source.revision_label}",
        }
        for model in registered_models(root, backend="lingbot_va")
    )
    camera_base = f"http://127.0.0.1:{system.web_preview.port}"
    return {
        "product": {"brand": "GALAXEA A1", "title": "Control"},
        "cameras": [
            {"id": "agent", "label": "Agent", "url": f"{camera_base}/agent.mjpg"},
            {"id": "wrist", "label": "Wrist", "url": f"{camera_base}/wrist.mjpg"},
        ],
        "camera_controls": [
            {
                "label": "Start cameras",
                "workflow": "camera",
                "values": {"action": "start"},
            },
            {
                "label": "Stop cameras",
                "workflow": "camera",
                "values": {"action": "stop"},
                "confirm": "Stop the persistent read-only camera monitor?",
            },
        ],
        "workflows": _workflow_forms(
            teleop_options=teleop_options,
            deployment_options=deployment_options,
            task_options=task_options,
            batch_options=batch_options,
            model_options=model_options,
            reset_options=reset_options,
        ),
        "config_types": config_store.catalog(),
        "configuration_groups": [
            {"label": "Teleop", "items": teleop_options},
            {"label": "Evaluation", "items": deployment_options},
            {"label": "Batch", "items": batch_options},
            {"label": "Reset", "items": reset_options},
            {"label": "Models", "items": model_options[1:]},
        ],
    }


def _workflow_forms(
    *,
    teleop_options: list[dict[str, str]],
    deployment_options: list[dict[str, str]],
    task_options: list[dict[str, str]],
    batch_options: list[dict[str, str]],
    model_options: list[dict[str, str]],
    reset_options: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "collect",
            "label": "Collect",
            "eyebrow": "DATA COLLECTION",
            "title": "Collect episodes",
            "submit_label": "Start collection",
            "fields": [
                _select_field(
                    "config",
                    "Teleop config",
                    teleop_options,
                    default=TELEOP_CONFIG.as_posix(),
                ),
                _text_field(
                    "experiment", "Experiment", placeholder="fruit_placement_v1"
                ),
                _text_field("task", "Task", placeholder="put the fruit into the bowl"),
            ],
        },
        {
            "id": "evaluate",
            "label": "Evaluation",
            "eyebrow": "LIVE EVALUATION",
            "title": "Run one evaluation",
            "submit_label": "Start evaluation",
            "fields": [
                _select_field(
                    "config",
                    "Deployment config",
                    deployment_options,
                    default=LINGBOT_CONFIG.as_posix(),
                ),
                _select_field("model", "Model", model_options, required=False),
                _select_field("task", "Task", task_options, depends_on="config"),
                _text_field(
                    "scene_note",
                    "Scene note",
                    placeholder="table centered, normal lighting",
                ),
            ],
        },
        {
            "id": "batch",
            "label": "Batch",
            "eyebrow": "BATCH",
            "title": "Run a repository plan",
            "submit_label": "Start batch",
            "fields": [
                _select_field(
                    "config",
                    "Batch config",
                    batch_options,
                    default=LINGBOT_BATCH_CONFIG.as_posix(),
                ),
                _select_field("model", "Model", model_options, required=False),
                _text_field(
                    "scene_note",
                    "Scene note",
                    placeholder="randomized setup A",
                ),
                {
                    "name": "resume",
                    "label": "Resume completed plan",
                    "type": "checkbox",
                    "default": False,
                },
            ],
        },
        {
            "id": "reset",
            "label": "Reset",
            "eyebrow": "RESET",
            "title": "Reset A1",
            "submit_label": "Run reset",
            "confirm": "Run the selected repository A1 reset now?",
            "fields": [
                _select_field(
                    "pose",
                    "Pose config",
                    reset_options,
                    default=A1_RESET_POSE.as_posix(),
                )
            ],
        },
    ]


def _select_field(
    name: str,
    label: str,
    options: list[dict[str, str]],
    *,
    default: str | None = None,
    required: bool = True,
    depends_on: str | None = None,
) -> dict[str, Any]:
    field: dict[str, Any] = {
        "name": name,
        "label": label,
        "type": "select",
        "required": required,
        "options": options,
    }
    if default is not None:
        field["default"] = default
    if depends_on is not None:
        field["depends_on"] = depends_on
    return field


def _text_field(name: str, label: str, *, placeholder: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "type": "text",
        "required": True,
        "placeholder": placeholder,
    }


def _option(path: Path, root: Path) -> dict[str, str]:
    return {"value": _reference(path, root), "label": path.stem}


def _reference(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
