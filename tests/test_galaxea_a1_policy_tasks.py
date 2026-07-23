import io
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.task_selection import (
    TaskSelectionCancelled,
    select_task,
)
from embodied_ops import (
    load_task_catalog,
)


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs/tasks/fruit_placement/catalog.json"


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
