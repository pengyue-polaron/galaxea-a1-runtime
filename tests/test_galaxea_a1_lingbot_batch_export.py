from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.lingbot.batch_config import load_lingbot_batch_config
from galaxea_a1_runtime.apps.lingbot.batch_export import (
    export_valid_lingbot_batch,
    render_lingbot_batch_report,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/runs/lingbot/fruit_placement.toml"
DEPLOYMENT = load_lingbot_batch_config(CONFIG, repo_root=REPO).deployment
MODEL_ID = DEPLOYMENT.policy_server.model.model_id
MODEL_REVISION = DEPLOYMENT.policy_server.model.source.revision


def _write_valid_run(
    root: Path,
    *,
    sequence: int,
    run_id: str | None = None,
) -> None:
    attempts = 3
    tasks = (
        "banana_blue_plate",
        "banana_bowl",
        "lemon_blue_plate",
        "red_mango_blue_plate",
        "red_mango_bowl",
        "lemon_bowl",
    )
    task_position = (sequence - 1) // attempts + 1
    attempt = (sequence - 1) % attempts + 1
    identity = run_id or f"valid-run-{sequence:02d}"
    run_dir = root / identity
    run_dir.mkdir(parents=True)
    video = "evaluation.mp4"
    (run_dir / video).write_bytes(f"video-{sequence}".encode())
    (run_dir / "runtime.log").write_text(f"runtime-{sequence}\n")
    (run_dir / "policy_server.log").write_text(f"policy-{sequence}\n")
    metadata = {
        "schema_version": 2,
        "frames": 1,
        "run": {
            "run_id": identity,
            "scene_note": "randomized_A",
            "status": "completed",
            "configuration": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            },
            "task": {"id": tasks[task_position - 1]},
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
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def test_batch_report_lists_exact_valid_and_pending_slots(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    recordings = tmp_path / "recordings"
    _write_valid_run(recordings, sequence=1)

    report = render_lingbot_batch_report(
        config,
        scene_note="randomized_A",
        recording_root=recordings,
    )

    assert "Valid slots: 1/18" in report
    assert "Pending slots: 17" in report
    assert "01/18: task=banana_blue_plate attempt=1/3 VALID" in report
    assert "02/18: task=banana_blue_plate attempt=2/3 PENDING" in report


def test_complete_batch_exports_manifest_video_metadata_and_logs(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    recordings = tmp_path / "recordings"
    for sequence in range(1, 19):
        _write_valid_run(recordings, sequence=sequence)

    archive_path = export_valid_lingbot_batch(
        config,
        scene_note="randomized_A",
        recording_root=recordings,
        export_root=tmp_path / "exports",
        now=datetime(2026, 7, 19, 2, 3, 4, tzinfo=timezone.utc),
    )

    assert archive_path.name == (
        "fruit-placement-scripted__randomized_A__20260719_020304.tar"
    )
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.load(manifest_file)
    assert len(names) == 73
    assert manifest["scene_note"] == "randomized_A"
    assert manifest["total_slots"] == 18
    assert len(manifest["runs"]) == 18
    assert manifest["runs"][0]["task"] == {
        "id": "banana_blue_plate",
        "prompt": "Put the banana into the blue plate",
        "distribution": "train",
    }
    assert len(manifest["runs"][0]["files"]) == 4
    assert all(len(item["sha256"]) == 64 for item in manifest["runs"][0]["files"])


def test_batch_export_rejects_pending_or_duplicate_valid_slots(tmp_path: Path):
    config = load_lingbot_batch_config(CONFIG, repo_root=REPO)
    recordings = tmp_path / "recordings"
    _write_valid_run(recordings, sequence=1)
    with pytest.raises(ValueError, match="incomplete.*pending slots: 2,3"):
        export_valid_lingbot_batch(
            config,
            scene_note="randomized_A",
            recording_root=recordings,
            export_root=tmp_path / "exports",
        )

    _write_valid_run(recordings, sequence=1, run_id="duplicate-valid-run-01")
    with pytest.raises(ValueError, match="duplicate valid slots: 1"):
        export_valid_lingbot_batch(
            config,
            scene_note="randomized_A",
            recording_root=recordings,
            export_root=tmp_path / "exports",
        )
