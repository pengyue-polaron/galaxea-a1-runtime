from pathlib import Path

from galaxea_a1_runtime.lerobot.migration import (
    LegacyDatasetKind,
    plan_raw_episodes_to_v30,
    plan_v21_to_v30,
)


def test_plan_v21_to_v30_uses_official_lerobot_converter():
    plan = plan_v21_to_v30(repo_id="galaxea/a1_task", lerobot_python="python3")

    assert plan.kind == LegacyDatasetKind.LEROBOT_V21
    assert "lerobot.scripts.convert_dataset_v21_to_v30" in plan.command
    assert "--repo-id=galaxea/a1_task" in plan.command


def test_plan_raw_episodes_to_v30_points_to_new_runtime_converter():
    plan = plan_raw_episodes_to_v30(
        source_root=Path("/data/raw/task"),
        target_repo_id="galaxea/a1_task",
        target_root=Path("/data/processed/a1_task"),
    )

    assert plan.kind == LegacyDatasetKind.RAW_EPISODES
    assert "galaxea_a1_runtime.lerobot.convert_raw" in plan.command
