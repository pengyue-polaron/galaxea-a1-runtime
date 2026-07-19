from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.run_artifacts import (
    LingBotBatchAttempt,
    finalize_lingbot_run,
    inspect_lingbot_run,
    lingbot_video_filename,
    prepare_lingbot_run,
    record_lingbot_evaluation_decision,
    record_lingbot_run_outcome,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/deployments/lingbot/fruit_placement_eef.toml"


def _prepare(tmp_path: Path, run_id: str = "20260718_010203_000000_red_mango_bowl"):
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    task = config.task_catalog.task("red_mango_bowl")
    started = datetime(2026, 7, 18, 1, 2, 3, tzinfo=timezone.utc)
    return prepare_lingbot_run(
        config,
        task,
        repo_root=REPO,
        scene_note="桌面偏左，蓝色盘子靠前",
        batch_attempt=LingBotBatchAttempt(
            batch_id="fruit-placement-scripted",
            task_position=5,
            task_count=6,
            attempt=2,
            attempt_count=3,
            sequence=14,
            total=18,
        ),
        output_root=tmp_path,
        run_id=run_id,
        now=started,
        git_commit="a" * 40,
        git_worktree_dirty=False,
    )


def test_lingbot_run_finalizes_video_prompt_metadata_and_both_logs(tmp_path: Path):
    paths = _prepare(tmp_path)
    paths.raw_runtime_log.write_text(
        "\x1b[1;32m[PASS]\x1b[0m started\r[RUN] call 1\r\nfinished\r\n"
    )
    paths.policy_server_log.write_text("server ready\n")
    paths.final_dir.mkdir()
    (paths.final_dir / paths.video_filename).write_bytes(b"video")
    (paths.final_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": paths.video_filename,
                "frames": 42,
            }
        )
    )

    final_dir = finalize_lingbot_run(
        tmp_path,
        paths.run_id,
        exit_code=0,
        now=datetime(2026, 7, 18, 1, 3, 4, tzinfo=timezone.utc),
    )

    assert final_dir == paths.final_dir
    assert (final_dir / "runtime.log").read_text() == (
        "[PASS] started\n[RUN] call 1\nfinished\n"
    )
    assert (final_dir / "policy_server.log").read_text() == "server ready\n"
    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["video"] == paths.video_filename
    assert metadata["frames"] == 42
    assert metadata["run"]["task"] == {
        "id": "red_mango_bowl",
        "prompt": "put the red mango into the bowl",
        "distribution": "train",
    }
    assert metadata["run"]["scene_note"] == "桌面偏左，蓝色盘子靠前"
    assert metadata["run"]["video_filename"] == paths.video_filename
    assert metadata["run"]["batch"] == {
        "id": "fruit-placement-scripted",
        "task_position": 5,
        "task_count": 6,
        "attempt": 2,
        "attempt_count": 3,
        "sequence": 14,
        "total": 18,
    }
    assert metadata["run"]["configuration"]["deployment"] == (
        "configs/deployments/lingbot/fruit_placement_eef.toml"
    )
    assert set(metadata["run"]["configuration"]["sha256"]) == {
        "deployment",
        "system",
        "task_catalog",
        "model",
        "model_contract",
        "model_manifest",
    }
    assert all(
        len(value) == 64
        for value in metadata["run"]["configuration"]["sha256"].values()
    )
    assert metadata["run"]["git"] == {
        "commit": "a" * 40,
        "worktree_dirty": False,
    }
    assert metadata["run"]["status"] == "completed"
    assert metadata["run"]["exit_code"] == 0
    assert metadata["artifacts"] == {
        "video": paths.video_filename,
        "runtime_log": "runtime.log",
        "policy_server_log": "policy_server.log",
        "incomplete_video_staging": False,
    }
    assert not paths.log_staging_dir.exists()
    assert finalize_lingbot_run(tmp_path, paths.run_id, exit_code=0) == final_dir


def test_failed_lingbot_start_still_preserves_metadata_and_logs(tmp_path: Path):
    paths = _prepare(tmp_path, "20260718_010203_000001_red_mango_bowl")
    paths.raw_runtime_log.write_text("startup failed\n")
    video_staging = tmp_path / f".{paths.run_id}.staging"
    video_staging.mkdir()

    final_dir = finalize_lingbot_run(tmp_path, paths.run_id, exit_code=2)

    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["run"]["status"] == "failed"
    assert metadata["artifacts"]["video"] is None
    assert metadata["artifacts"]["incomplete_video_staging"] is True
    assert (final_dir / "runtime.log").read_text() == "startup failed\n"
    assert (final_dir / "policy_server.log").is_file()
    assert video_staging.is_dir()


def test_ik_rejection_requires_a_persisted_operator_evaluation_decision(
    tmp_path: Path,
):
    paths = _prepare(tmp_path, "20260718_010203_000003_red_mango_bowl")
    paths.raw_runtime_log.write_text("IK target stopped\n")
    paths.final_dir.mkdir()
    (paths.final_dir / paths.video_filename).write_bytes(b"video")
    record_lingbot_run_outcome(
        tmp_path,
        paths.run_id,
        kind="ik_target_rejected",
        message="solution delta 1.62 > 1.50 rad",
    )

    final_dir = finalize_lingbot_run(tmp_path, paths.run_id, exit_code=0)

    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["run"]["status"] == "safety_stopped"
    assert metadata["run"]["termination"] == {
        "kind": "ik_target_rejected",
        "message": "solution delta 1.62 > 1.50 rad",
    }
    result = inspect_lingbot_run(tmp_path, paths.run_id)
    assert result.status == "safety_stopped"
    assert result.evaluation_decision == ""

    decision_path = record_lingbot_evaluation_decision(
        tmp_path,
        paths.run_id,
        decision="counted",
        now=datetime(2026, 7, 18, 1, 4, 5, tzinfo=timezone.utc),
    )

    decided = json.loads(decision_path.read_text())
    assert decided["run"]["evaluation"] == {
        "decision": "counted",
        "decided_at": "2026-07-18T01:04:05+00:00",
    }
    result = inspect_lingbot_run(tmp_path, paths.run_id)
    assert result.evaluation_decision == "counted"
    assert "RUN_STATUS=safety_stopped" in result.shell()
    assert "RUN_EVALUATION_DECISION=counted" in result.shell()
    assert not paths.log_staging_dir.exists()


def test_operator_interrupt_marker_is_not_counted_as_completed(tmp_path: Path):
    paths = _prepare(tmp_path, "20260718_010203_000004_red_mango_bowl")
    record_lingbot_run_outcome(
        tmp_path,
        paths.run_id,
        kind="operator_interrupted",
        message="operator stopped run",
    )

    final_dir = finalize_lingbot_run(tmp_path, paths.run_id, exit_code=0)

    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["run"]["status"] == "interrupted"
    assert metadata["run"]["exit_code"] == 0


def test_workspace_rejection_can_be_discarded_for_retry(tmp_path: Path):
    paths = _prepare(tmp_path, "20260718_010203_000005_red_mango_bowl")
    record_lingbot_run_outcome(
        tmp_path,
        paths.run_id,
        kind="workspace_target_rejected",
        message="target x=0.472 exceeds max=0.47",
    )

    final_dir = finalize_lingbot_run(tmp_path, paths.run_id, exit_code=0)

    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["run"]["status"] == "safety_stopped"
    assert metadata["run"]["termination"]["kind"] == ("workspace_target_rejected")

    record_lingbot_evaluation_decision(
        tmp_path,
        paths.run_id,
        decision="discarded",
    )

    assert inspect_lingbot_run(tmp_path, paths.run_id).evaluation_decision == (
        "discarded"
    )


def test_legacy_target_rejection_accepts_an_operator_decision(tmp_path: Path):
    paths = _prepare(tmp_path, "20260718_010203_000006_red_mango_bowl")
    paths.raw_runtime_log.write_text(
        "ValueError: EEF policy action is outside the configured workspace on x\n"
    )
    final_dir = finalize_lingbot_run(tmp_path, paths.run_id, exit_code=1)

    record_lingbot_evaluation_decision(
        tmp_path,
        paths.run_id,
        decision="discarded",
    )

    metadata = json.loads((final_dir / "metadata.json").read_text())
    assert metadata["run"]["status"] == "failed"
    assert metadata["run"]["evaluation"]["decision"] == "discarded"


def test_prepared_run_shell_values_share_one_run_identity(tmp_path: Path):
    paths = _prepare(tmp_path, "20260718_010203_000002_red_mango_bowl")

    rendered = paths.shell()

    assert f"RUN_ID={paths.run_id}" in rendered
    assert f"RUN_FINAL_DIR={paths.final_dir}" in rendered
    assert f"RUN_RUNTIME_RAW_LOG={paths.raw_runtime_log}" in rendered
    assert f"RUN_POLICY_LOG={paths.policy_server_log}" in rendered
    assert "RUN_VIDEO_FILENAME=" in rendered
    assert paths.video_filename in rendered


def test_lingbot_video_name_is_scene_prompt_date_and_portable():
    now = datetime(2026, 7, 18, 1, 2, 3, tzinfo=timezone.utc)

    filename = lingbot_video_filename(
        "桌面偏左 / 光照正常",
        "put the red mango into the bowl",
        now=now,
    )

    assert filename == (
        "桌面偏左_光照正常__put_the_red_mango_into_the_bowl__20260718_010203.mp4"
    )
    assert "/" not in filename
    long_filename = lingbot_video_filename("场" * 120, "动" * 200)
    assert len(long_filename.encode("utf-8")) <= 240
