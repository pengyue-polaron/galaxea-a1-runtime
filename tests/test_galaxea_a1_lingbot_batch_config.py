from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.lingbot.batch_config import (
    bash_config,
    load_lingbot_batch_config,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/runs/lingbot/fruit_placement.toml"
MANGO_CONFIG = REPO / "configs/runs/lingbot/mango_placement.toml"


def _copy(tmp_path: Path) -> Path:
    path = tmp_path / "batch.toml"
    path.write_text(CONFIG.read_text())
    return path


def test_lingbot_batch_plan_composes_tasks_retries_and_shared_a1_reset():
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)

    assert config.batch_id == "fruit-placement-scripted"
    assert config.retries_per_prompt == 2
    assert config.attempts_per_prompt == 3
    assert config.total_attempts == 18
    assert config.reset_pose == REPO / "configs/poses/a1_collection_start.toml"
    assert config.task_ids[0] == "banana_blue_plate"
    assert config.task_ids[-1] == "lemon_bowl"
    shell = bash_config(config)
    assert "BATCH_TOTAL_ATTEMPTS=18" in shell
    assert "BATCH_RETRIES_PER_PROMPT=2" in shell


def test_mango_model_batch_plan_runs_its_curated_task_suite():
    config = load_lingbot_batch_config(
        MANGO_CONFIG,
        repo_root=REPO,
        model_selector="mango_placement_eef",
    )

    assert config.deployment.policy_server.model.model_id == (
        "lingbot/a1_mango_placement_eef"
    )
    catalog_task_ids = {task.task_id for task in config.deployment.task_catalog.tasks}
    assert set(config.task_ids) <= catalog_task_ids
    assert config.attempts_per_prompt == 3
    assert config.total_attempts == len(config.task_ids) * 3


def test_lingbot_batch_plan_rejects_unknown_or_duplicate_tasks(tmp_path: Path):
    unknown = _copy(tmp_path)
    unknown.write_text(unknown.read_text().replace('"lemon_bowl",', '"unknown_task",'))
    with pytest.raises(ValueError, match="unknown task id"):
        load_lingbot_batch_config(unknown, repo_root=REPO)

    duplicate = _copy(tmp_path)
    duplicate.write_text(
        duplicate.read_text().replace('"lemon_bowl",', '"banana_bowl",')
    )
    with pytest.raises(ValueError, match="duplicates"):
        load_lingbot_batch_config(duplicate, repo_root=REPO)


def test_lingbot_batch_retry_count_must_be_non_negative(tmp_path: Path):
    path = _copy(tmp_path)
    path.write_text(
        path.read_text().replace("retries_per_prompt = 2", "retries_per_prompt = -1")
    )

    with pytest.raises(ValueError, match="non-negative"):
        load_lingbot_batch_config(path, repo_root=REPO)


def test_lingbot_batch_retry_count_has_no_arbitrary_upper_cap(tmp_path: Path):
    path = _copy(tmp_path)
    path.write_text(
        path.read_text().replace("retries_per_prompt = 2", "retries_per_prompt = 250")
    )

    config = load_lingbot_batch_config(path, repo_root=REPO)

    assert config.attempts_per_prompt == 251
    assert config.total_attempts == 1506
