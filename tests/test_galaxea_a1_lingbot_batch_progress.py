from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.batch_progress import (
    inspect_lingbot_batch_progress,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/runs/lingbot/fruit_placement.toml"
DEPLOYMENT = load_lingbot_batch_config(CONFIG, repo_root=REPO).deployment
MODEL_ID = DEPLOYMENT.policy_server.model.model_id
MODEL_REVISION = DEPLOYMENT.policy_server.model.source.revision


def _write_run(
    root: Path,
    *,
    name: str,
    sequence: int,
    status: str,
    scene_note: str = "randomized_A",
    runtime_log: str = "run finished\n",
    task_id: str | None = None,
    evaluation_decision: str | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> None:
    attempts = 3
    task_position = (sequence - 1) // attempts + 1
    attempt = (sequence - 1) % attempts + 1
    tasks = (
        "banana_blue_plate",
        "banana_bowl",
        "lemon_blue_plate",
        "red_mango_blue_plate",
        "red_mango_bowl",
        "lemon_bowl",
    )
    run_dir = root / name
    run_dir.mkdir()
    video = "recording.mp4"
    (run_dir / video).write_bytes(b"video")
    (run_dir / "runtime.log").write_text(runtime_log)
    (run_dir / "policy_server.log").write_text("server\n")
    metadata = {
        "schema_version": 2,
        "frames": 10,
        "run": {
            "run_id": name,
            "scene_note": scene_note,
            "status": status,
            "configuration": {
                "model_id": model_id,
                "model_revision": model_revision,
            },
            "task": {"id": task_id or tasks[task_position - 1]},
            "batch": {
                "id": "fruit-placement-scripted",
                "task_position": task_position,
                "task_count": 6,
                "attempt": attempt,
                "attempt_count": attempts,
                "sequence": sequence,
                "total": 18,
            },
        },
        "artifacts": {
            "video": video,
            "runtime_log": "runtime.log",
            "policy_server_log": "policy_server.log",
        },
    }
    if evaluation_decision is not None:
        metadata["run"]["evaluation"] = {
            "decision": evaluation_decision,
            "decided_at": "2026-07-19T01:00:00+08:00",
        }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def test_resume_skips_completed_safety_stopped_and_legacy_ik_slots(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    _write_run(
        tmp_path,
        name="legacy-ik",
        sequence=1,
        status="failed",
        runtime_log=(
            "RuntimeError: A1 EEF IK solution exceeds the configured joint delta: "
            "1.622263 > 1.500000 rad\n"
        ),
        evaluation_decision="counted",
    )
    _write_run(
        tmp_path,
        name="safe-stop",
        sequence=2,
        status="safety_stopped",
        evaluation_decision="counted",
    )
    _write_run(tmp_path, name="complete", sequence=3, status="completed")
    _write_run(
        tmp_path,
        name="legacy-workspace",
        sequence=4,
        status="failed",
        runtime_log=(
            "ValueError: EEF policy action is outside the configured workspace on x\n"
        ),
        evaluation_decision="counted",
    )
    _write_run(
        tmp_path,
        name="infrastructure-failure",
        sequence=5,
        status="failed",
        runtime_log="serial disconnected\n",
    )
    _write_run(tmp_path, name="interrupted", sequence=6, status="interrupted")
    _write_run(
        tmp_path,
        name="different-scene",
        sequence=7,
        status="completed",
        scene_note="randomized_B",
    )

    progress = inspect_lingbot_batch_progress(
        config,
        scene_note="randomized_A",
        output_root=tmp_path,
    )

    assert progress.completed_sequences == (1, 2, 3, 4)
    assert progress.legacy_safety_stop_sequences == (1, 4)
    assert progress.completed_count == 4
    assert progress.pending_count == 14
    assert "BATCH_COMPLETED_SEQUENCES_CSV=1,2,3,4" in progress.shell()


def test_resume_requires_valid_artifacts_and_exact_task_slot(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    _write_run(
        tmp_path,
        name="wrong-task",
        sequence=1,
        status="completed",
        task_id="banana_bowl",
    )
    _write_run(tmp_path, name="missing-video", sequence=2, status="completed")
    (tmp_path / "missing-video" / "recording.mp4").unlink()
    _write_run(tmp_path, name="zero-frames", sequence=3, status="completed")
    metadata_path = tmp_path / "zero-frames" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["frames"] = 0
    metadata_path.write_text(json.dumps(metadata))
    _write_run(tmp_path, name="undecided", sequence=4, status="safety_stopped")
    _write_run(
        tmp_path,
        name="discarded",
        sequence=5,
        status="safety_stopped",
        evaluation_decision="discarded",
    )
    _write_run(
        tmp_path,
        name="legacy-discarded",
        sequence=6,
        status="failed",
        runtime_log="A1 EEF IK did not converge: iterations=200\n",
        evaluation_decision="discarded",
    )

    progress = inspect_lingbot_batch_progress(
        config,
        scene_note="randomized_A",
        output_root=tmp_path,
    )

    assert progress.completed_sequences == ()
    assert progress.pending_count == 18


def test_resume_requires_the_selected_model_identity(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    _write_run(
        tmp_path,
        name="different-model-id",
        sequence=1,
        status="completed",
        model_id="lingbot/a1_mango_placement_eef",
    )
    _write_run(
        tmp_path,
        name="different-model-revision",
        sequence=2,
        status="completed",
        model_revision="0" * 40,
    )
    _write_run(tmp_path, name="selected-model", sequence=3, status="completed")

    progress = inspect_lingbot_batch_progress(
        config,
        scene_note="randomized_A",
        output_root=tmp_path,
    )

    assert progress.completed_sequences == (3,)
