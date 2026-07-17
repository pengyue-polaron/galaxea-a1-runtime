import io
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.task_selection import (
    TaskSelectionCancelled,
    select_task,
)
from galaxea_a1_runtime.configuration.tasks import load_task_catalog


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs/tasks/fruit_placement.toml"


def test_tracked_task_catalog_exposes_training_and_ood_prompts():
    catalog = load_task_catalog(CATALOG, repo_root=REPO)

    assert catalog.catalog_id == "fruit-placement-v2"
    assert [task.prompt for task in catalog.tasks] == [
        "Put the banana into the blue plate",
        "put the banana into the bowl",
        "put the lemon into the blue plate",
        "put the red mango into the blue plate",
        "put the red mango into the bowl",
        "put the lemon into the bowl",
    ]
    assert [task.task_id for task in catalog.tasks] == [
        "banana_blue_plate",
        "banana_bowl",
        "lemon_blue_plate",
        "red_mango_blue_plate",
        "red_mango_bowl",
        "lemon_bowl",
    ]
    assert [task.distribution for task in catalog.tasks] == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "ood",
    ]


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
    text = CATALOG.read_text().replace(
        'prompt = "put the banana into the bowl"',
        'prompt = "Put the banana into the blue plate"',
    )
    path = tmp_path / "tasks.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="prompts must be unique"):
        load_task_catalog(path, repo_root=REPO)


def test_task_catalog_rejects_unknown_distribution(tmp_path):
    path = tmp_path / "tasks.toml"
    path.write_text(
        CATALOG.read_text().replace('distribution = "ood"', 'distribution = "test"')
    )

    with pytest.raises(ValueError, match="distribution must be 'train' or 'ood'"):
        load_task_catalog(path, repo_root=REPO)
