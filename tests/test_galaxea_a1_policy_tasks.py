from pathlib import Path

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
