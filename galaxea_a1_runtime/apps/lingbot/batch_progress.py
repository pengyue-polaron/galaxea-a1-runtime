"""Inspect durable LingBot artifacts and identify resumable batch slots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.batch_config import (
    DEFAULT_BATCH_CONFIG,
    LingBotBatchConfig,
    load_lingbot_batch_config,
)
from galaxea_a1_runtime.apps.lingbot.operator_input import validate_scene_note
from galaxea_a1_runtime.configuration.base import shell_assign
from galaxea_a1_runtime.console import ArgumentParser


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LingBotValidBatchRun:
    sequence: int
    task_position: int
    attempt: int
    run_id: str
    run_dir: Path
    status: str
    evaluation_decision: str


@dataclass(frozen=True)
class LingBotBatchProgress:
    valid_runs: tuple[LingBotValidBatchRun, ...]
    total: int

    @property
    def completed_sequences(self) -> tuple[int, ...]:
        return tuple(sorted({run.sequence for run in self.valid_runs}))

    @property
    def duplicate_sequences(self) -> tuple[int, ...]:
        counts: dict[int, int] = {}
        for run in self.valid_runs:
            counts[run.sequence] = counts.get(run.sequence, 0) + 1
        return tuple(
            sorted(sequence for sequence, count in counts.items() if count > 1)
        )

    @property
    def completed_count(self) -> int:
        return len(self.completed_sequences)

    @property
    def pending_count(self) -> int:
        return self.total - self.completed_count

    def shell(self) -> str:
        values = (
            (
                "BATCH_COMPLETED_SEQUENCES_CSV",
                ",".join(str(value) for value in self.completed_sequences),
            ),
            ("BATCH_COMPLETED_COUNT", str(self.completed_count)),
            ("BATCH_PENDING_COUNT", str(self.pending_count)),
        )
        return "\n".join(shell_assign(name, value) for name, value in values)


def inspect_lingbot_batch_progress(
    config: LingBotBatchConfig,
    *,
    scene_note: str,
    output_root: Path | None = None,
) -> LingBotBatchProgress:
    """Return current-plan slots with complete, valid durable artifacts."""

    note = validate_scene_note(scene_note)
    recordings = (output_root or config.deployment.recording.output_root).resolve()
    valid_runs: list[LingBotValidBatchRun] = []
    if recordings.is_dir():
        for metadata_path in recordings.glob("*/metadata.json"):
            classification = _classify_run(
                metadata_path, config=config, scene_note=note
            )
            if classification is None:
                continue
            valid_runs.append(classification)
    return LingBotBatchProgress(
        valid_runs=tuple(
            sorted(valid_runs, key=lambda run: (run.sequence, run.run_id))
        ),
        total=config.total_attempts,
    )


def _classify_run(
    metadata_path: Path,
    *,
    config: LingBotBatchConfig,
    scene_note: str,
) -> LingBotValidBatchRun | None:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    run = metadata.get("run")
    artifacts = metadata.get("artifacts")
    if not isinstance(run, dict) or not isinstance(artifacts, dict):
        return None
    if run.get("scene_note") != scene_note:
        return None
    configuration = run.get("configuration")
    model = config.deployment.policy_server.model
    if not isinstance(configuration, dict) or (
        configuration.get("model_id") != model.model_id
        or configuration.get("model_revision") != model.source.revision
    ):
        return None
    batch = run.get("batch")
    task = run.get("task")
    if not isinstance(batch, dict) or not isinstance(task, dict):
        return None

    task_position = _plain_int(batch.get("task_position"))
    attempt = _plain_int(batch.get("attempt"))
    sequence = _plain_int(batch.get("sequence"))
    if task_position is None or attempt is None or sequence is None:
        return None
    if not 1 <= task_position <= len(config.task_ids):
        return None
    if not 1 <= attempt <= config.attempts_per_prompt:
        return None
    try:
        slot = config.plan.slot(task_position=task_position, attempt=attempt)
    except ValueError:
        return None
    expected_sequence = slot.sequence
    expected_batch = slot.to_dict()
    if any(
        batch.get(key, slot.task_id if key == "task_id" else None) != value
        for key, value in expected_batch.items()
    ):
        return None
    if sequence != expected_sequence or task.get("id") != slot.task_id:
        return None

    run_dir = metadata_path.parent
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or run_id != run_dir.name:
        return None
    if not _artifacts_are_complete(metadata, artifacts, run_dir):
        return None
    status = run.get("status")
    if status == "completed":
        return LingBotValidBatchRun(
            sequence=sequence,
            task_position=task_position,
            attempt=attempt,
            run_id=run_id,
            run_dir=run_dir,
            status=status,
            evaluation_decision="completed",
        )
    if status != "safety_stopped":
        return None
    evaluation = run.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("decision") == "counted":
        return LingBotValidBatchRun(
            sequence=sequence,
            task_position=task_position,
            attempt=attempt,
            run_id=run_id,
            run_dir=run_dir,
            status=status,
            evaluation_decision="counted",
        )
    return None


def _artifacts_are_complete(metadata: dict, artifacts: dict, run_dir: Path) -> bool:
    if artifacts.get("runtime_log") != "runtime.log":
        return False
    if artifacts.get("policy_server_log") != "policy_server.log":
        return False
    if not (run_dir / "runtime.log").is_file():
        return False
    if not (run_dir / "policy_server.log").is_file():
        return False
    video = artifacts.get("video")
    if not isinstance(video, str) or not video or Path(video).name != video:
        return False
    video_path = run_dir / video
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return False
    frames = _plain_int(metadata.get("frames"))
    return frames is not None and frames > 0


def _plain_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--scene-note", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_BATCH_CONFIG)
    args = parser.parse_args()
    config = load_lingbot_batch_config(
        args.config,
        repo_root=args.repo_root,
        model_selector=args.model,
    )
    progress = inspect_lingbot_batch_progress(
        config,
        scene_note=args.scene_note,
        output_root=args.output_root,
    )
    print(progress.shell() if args.shell else progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
