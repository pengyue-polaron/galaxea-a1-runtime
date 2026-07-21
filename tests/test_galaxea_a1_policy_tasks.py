import io
import json
import shutil
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.task_selection import (
    TaskSelectionCancelled,
    select_task,
)
from galaxea_a1_runtime.configuration.tasks import (
    load_task_catalog,
    register_task_prompt,
)


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs/tasks/fruit_placement/catalog.json"


def _catalog_copy(tmp_path: Path) -> Path:
    root = tmp_path / "fruit_placement"
    shutil.copytree(CATALOG.parent, root)
    return root / "catalog.json"


def test_tracked_task_catalog_exposes_training_and_ood_prompts():
    catalog = load_task_catalog(CATALOG, repo_root=REPO)

    assert catalog.catalog_id == "fruit-placement-v3"
    prompts = {task.task_id: (task.prompt, task.distribution) for task in catalog.tasks}
    assert {
        "banana_blue_plate": (
            "Put the banana into the blue plate",
            "train",
        ),
        "banana_bowl": ("put the banana into the bowl", "train"),
        "lemon_blue_plate": ("put the lemon into the blue plate", "train"),
        "red_mango_blue_plate": (
            "put the red mango into the blue plate",
            "train",
        ),
        "red_mango_bowl": ("put the red mango into the bowl", "train"),
        "lemon_bowl": ("put the lemon into the bowl", "ood"),
    }.items() <= prompts.items()


def test_task_selection_accepts_number_id_or_exact_tracked_prompt():
    catalog = load_task_catalog(CATALOG, repo_root=REPO)

    for answer, expected in (
        ("2", "banana_bowl"),
        ("lemon_blue_plate", "lemon_blue_plate"),
        ("put the red mango into the bowl", "red_mango_bowl"),
        ("6", "lemon_bowl"),
    ):
        output = io.StringIO()
        task = select_task(catalog, input_fn=lambda value=answer: value, output=output)
        assert task.task_id == expected
        assert "without starting model or hardware" in output.getvalue()
        assert "[lemon_bowl] [OOD]" in output.getvalue()


def test_task_selection_reprompts_unknown_values_and_supports_cancel():
    catalog = load_task_catalog(CATALOG, repo_root=REPO)
    answers = iter(("unknown", "1"))
    output = io.StringIO()

    task = select_task(catalog, input_fn=lambda: next(answers), output=output)

    assert task.task_id == "banana_blue_plate"
    assert "Unknown task" in output.getvalue()
    with pytest.raises(TaskSelectionCancelled, match="cancelled"):
        select_task(catalog, input_fn=lambda: "q", output=io.StringIO())


def test_task_catalog_rejects_duplicate_prompts(tmp_path):
    path = _catalog_copy(tmp_path)
    prompt_path = path.parent / "prompts/banana_bowl.json"
    data = json.loads(prompt_path.read_text())
    data["prompt"] = "Put the banana into the blue plate"
    prompt_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="prompts must be unique"):
        load_task_catalog(path, repo_root=REPO)


def test_task_catalog_rejects_unknown_distribution(tmp_path):
    path = _catalog_copy(tmp_path)
    prompt_path = path.parent / "prompts/lemon_bowl.json"
    data = json.loads(prompt_path.read_text())
    data["distribution"] = "test"
    prompt_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="distribution must be 'train' or 'ood'"):
        load_task_catalog(path, repo_root=REPO)


def test_task_registration_is_create_only_and_immediately_loadable(tmp_path):
    path = _catalog_copy(tmp_path)
    previous_orders = [
        json.loads(prompt_path.read_text())["order"]
        for prompt_path in (path.parent / "prompts").glob("*.json")
    ]

    created = register_task_prompt(
        path,
        task_id="green_apple_bowl",
        prompt="put the green apple into the bowl",
        distribution="ood",
    )

    assert created == path.parent / "prompts/green_apple_bowl.json"
    payload = json.loads(created.read_text())
    assert payload == {
        "schema_version": 1,
        "order": max(previous_orders) + 10,
        "id": "green_apple_bowl",
        "prompt": "put the green apple into the bowl",
        "distribution": "ood",
    }
    assert load_task_catalog(path).task("green_apple_bowl").distribution == "ood"
    with pytest.raises(FileExistsError, match="already registered"):
        register_task_prompt(
            path,
            task_id="green_apple_bowl",
            prompt="a different prompt",
            distribution="ood",
        )
    with pytest.raises(ValueError, match="prompt is already registered"):
        register_task_prompt(
            path,
            task_id="green_apple_bowl_again",
            prompt="put the green apple into the bowl",
            distribution="ood",
        )
