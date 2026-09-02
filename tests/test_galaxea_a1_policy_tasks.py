import json
import shutil
from pathlib import Path

from embodied_ops import (
    load_task_catalog,
)

from galaxea_a1_runtime.cli import main as cli_main


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


def test_prompt_cli_registers_and_lists_a_validated_prompt(tmp_path, capsys):
    shutil.copytree(REPO / "configs/tasks", tmp_path / "configs/tasks")
    catalog = Path("configs/tasks/fruit_placement/catalog.json")

    assert (
        cli_main(
            [
                "prompt",
                "register",
                str(catalog),
                "green_apple_bowl",
                "put the green apple into the bowl",
                "--distribution",
                "ood",
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        cli_main(
            [
                "prompt",
                "list",
                str(catalog),
                "--json",
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    listed = json.loads(capsys.readouterr().out)

    assert listed["catalogs"][0]["prompts"][-1] == {
        "id": "green_apple_bowl",
        "prompt": "put the green apple into the bowl",
        "distribution": "ood",
    }


def test_prompt_cli_rejects_a_duplicate_task_id(tmp_path, capsys):
    shutil.copytree(REPO / "configs/tasks", tmp_path / "configs/tasks")

    result = cli_main(
        [
            "prompt",
            "register",
            "configs/tasks/fruit_placement/catalog.json",
            "banana_bowl",
            "a replacement prompt",
            "--distribution",
            "train",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert result == 2
    assert "already registered" in capsys.readouterr().err
