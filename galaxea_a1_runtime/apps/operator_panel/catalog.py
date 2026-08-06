"""Build the dynamic A1 form and configuration catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_ops.operator_panel import (
    PANEL_CATALOG_SCHEMA_VERSION,
    RepositoryDocumentStore,
    checkbox_field,
    combobox_field,
    order_workflow_forms,
    select_field,
    standard_camera_controls,
    standard_core_workflows,
    standard_panel_product,
    text_field,
)

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.configuration.paths import (
    A1_RESET_POSE,
    LINGBOT_BATCH_CONFIG,
    LINGBOT_CONFIG,
    SYSTEM_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from embodied_ops import load_task_catalog
from galaxea_a1_runtime.models.registry import registered_models
from galaxea_a1_runtime.teleop.config import load_teleop_config

from .configuration import looks_like_a1_pose


def build_a1_catalog(
    repo_root: Path, document_store: RepositoryDocumentStore
) -> dict[str, Any]:
    root = repo_root.resolve()
    system = load_system_config(root / SYSTEM_CONFIG, repo_root=root)
    teleop_options = []
    experiment_options = []
    for path in sorted((root / "configs/teleop").glob("*.toml")):
        config = load_teleop_config(path, repo_root=root)
        config_reference = _reference(config.path, root)
        teleop_options.append(_option(config.path, root))
        experiment_options.extend(
            _experiment_options(
                config.collection.dataset_root,
                depends_value=config_reference,
            )
        )

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
    prompt_catalog_options = []
    collection_task_options = []
    for path in sorted((root / "configs/tasks").glob("*/catalog.json")):
        task_catalog = load_task_catalog(path, repo_root=root)
        prompt_catalog_options.append(
            {
                "value": _reference(task_catalog.path, root),
                "label": task_catalog.catalog_id,
            }
        )
        collection_task_options.extend(
            {
                "value": task.prompt,
                "label": (
                    f"{task_catalog.catalog_id} · {task.task_id} · {task.prompt}"
                ),
            }
            for task in task_catalog.tasks
            if task.distribution == "train"
        )
    return {
        "schema_version": PANEL_CATALOG_SCHEMA_VERSION,
        "product": standard_panel_product("GALAXEA A1"),
        "cameras": [
            {
                "id": "agent",
                "label": "Agent",
                "port": system.web_preview.port,
                "path": "/agent.mjpg",
            },
            {
                "id": "wrist",
                "label": "Wrist",
                "port": system.web_preview.port,
                "path": "/wrist.mjpg",
            },
        ],
        "camera_controls": standard_camera_controls(
            stop_confirm="Stop the persistent read-only camera monitor?"
        ),
        "workflows": _workflow_forms(
            teleop_options=teleop_options,
            experiment_options=experiment_options,
            collection_task_options=collection_task_options,
            deployment_options=deployment_options,
            task_options=task_options,
            batch_options=batch_options,
            model_options=model_options,
            reset_options=reset_options,
        ),
        "registrations": _registration_forms(prompt_catalog_options),
        "configuration_types": document_store.catalog(),
        "configuration_groups": [
            {"label": "Teleop", "items": teleop_options},
            {"label": "Evaluation", "items": deployment_options},
            {"label": "Batch", "items": batch_options},
            {"label": "Reset", "items": reset_options},
            {"label": "Models", "items": model_options[1:]},
        ],
    }


def _registration_forms(
    prompt_catalog_options: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "prompt",
            "label": "Prompts",
            "eyebrow": "PROMPT REGISTRY",
            "title": "Register a prompt",
            "description": (
                "Create one validated prompt record without modifying existing entries."
            ),
            "submit_label": "Register prompt",
            "confirm": "Register this prompt in the repository?",
            "fields": [
                select_field("catalog", "Catalog", prompt_catalog_options),
                text_field(
                    "prompt",
                    "Prompt",
                    placeholder="put the green apple into the bowl",
                ),
                {
                    **text_field("task_id", "Task ID", placeholder="green_apple_bowl"),
                    "derive_from": "prompt",
                    "transform": "snake_case",
                },
                select_field(
                    "distribution",
                    "Distribution",
                    [
                        {"value": "ood", "label": "OOD"},
                        {"value": "train", "label": "Train"},
                    ],
                    default="ood",
                ),
            ],
        }
    ]


def _workflow_forms(
    *,
    teleop_options: list[dict[str, str]],
    experiment_options: list[dict[str, str]],
    collection_task_options: list[dict[str, str]],
    deployment_options: list[dict[str, str]],
    task_options: list[dict[str, str]],
    batch_options: list[dict[str, str]],
    model_options: list[dict[str, str]],
    reset_options: list[dict[str, str]],
) -> list[dict[str, Any]]:
    collection_config = select_field(
        "config",
        "Collection config",
        teleop_options,
        default=TELEOP_CONFIG.as_posix(),
    )
    dataset_fields = _dataset_fields(teleop_options, experiment_options)
    core = standard_core_workflows(
        hardware_fields=[collection_config],
        collect_fields=[
            collection_config,
            combobox_field(
                "experiment",
                "Experiment",
                experiment_options,
                placeholder="fruit_placement_v1",
                help_text=(
                    "Select an existing experiment to append episodes, or type "
                    "a new name to create one."
                ),
                depends_on="config",
            ),
            combobox_field(
                "task",
                "Task prompt",
                collection_task_options,
                placeholder="pick up the object and place it at the target",
                help_text=(
                    "Select a tracked training prompt or type an exact new prompt. "
                    "One experiment may contain episodes from multiple prompts."
                ),
            ),
        ],
        reset_fields=[
            select_field(
                "pose",
                "Pose config",
                reset_options,
                default=A1_RESET_POSE.as_posix(),
            )
        ],
        dataset_fields=dataset_fields,
        reset_confirm=(
            "This moves A1 to the selected tracked collection pose. "
            "Confirm the workspace is clear?"
        ),
    )
    extensions = [
        {
            "id": "evaluate",
            "label": "Evaluation",
            "eyebrow": "LIVE EVALUATION",
            "title": "Run one evaluation",
            "description": "Run one tracked task and preserve its result.",
            "submit_label": "Start evaluation",
            "fields": [
                select_field(
                    "config",
                    "Deployment config",
                    deployment_options,
                    default=LINGBOT_CONFIG.as_posix(),
                ),
                select_field("model", "Model", model_options, required=False),
                select_field("task", "Task", task_options, depends_on="config"),
                text_field(
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
            "description": "Execute the selected tracked evaluation plan.",
            "submit_label": "Start batch",
            "fields": [
                select_field(
                    "config",
                    "Batch config",
                    batch_options,
                    default=LINGBOT_BATCH_CONFIG.as_posix(),
                ),
                select_field("model", "Model", model_options, required=False),
                text_field(
                    "scene_note",
                    "Scene note",
                    placeholder="randomized setup A",
                ),
                checkbox_field("resume", "Resume completed plan"),
            ],
        },
    ]
    return order_workflow_forms([*core, *extensions])


def _dataset_fields(
    teleop_options: list[dict[str, str]],
    experiment_options: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        select_field(
            "config",
            "Collection config",
            teleop_options,
            default=TELEOP_CONFIG.as_posix(),
        ),
        combobox_field(
            "experiment",
            "Experiment",
            experiment_options,
            placeholder="fruit_placement_v1",
            help_text="Select an existing canonical dataset or enter its exact name.",
            depends_on="config",
        ),
    ]


def _experiment_options(
    dataset_root: Path, *, depends_value: str
) -> list[dict[str, str]]:
    if not dataset_root.is_dir():
        return []
    options = []
    for path in sorted(dataset_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            experiment = validate_experiment_name(path.name)
        except ValueError:
            continue
        options.append(
            {
                "value": experiment,
                "label": experiment,
                "depends_value": depends_value,
            }
        )
    return options


def _option(path: Path, root: Path) -> dict[str, str]:
    return {"value": _reference(path, root), "label": path.stem}


def _reference(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
