"""Per-run LingBot video, metadata, and log artifact lifecycle."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from embodied_ops.artifacts import atomic_write_json, file_sha256
from embodied_ops.evaluation import EvaluationSlot
from embodied_ops.operator_panel import strip_protocol_events

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.operator_input import validate_scene_note
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.configuration.base import shell_assign
from embodied_ops import TaskPrompt
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.hardware.video_recorder import recording_run_id


ROOT = Path(__file__).resolve().parents[3]
_CONTEXT_NAME = "context.json"
_OUTCOME_NAME = "outcome.json"
_RAW_RUNTIME_LOG_NAME = "runtime.terminal.log"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OUTCOME_KINDS = frozenset(
    {"ik_target_rejected", "workspace_target_rejected", "operator_interrupted"}
)
_EVALUATION_DECISIONS = frozenset({"counted", "discarded"})


@dataclass(frozen=True)
class LingBotRunPaths:
    run_id: str
    output_root: Path
    final_dir: Path
    log_staging_dir: Path
    raw_runtime_log: Path
    policy_server_log: Path
    front_video_filename: str = ""
    wrist_video_filename: str = ""

    def shell(self) -> str:
        values = (
            ("RUN_ID", self.run_id),
            ("RUN_LOG_STAGING_DIR", str(self.log_staging_dir)),
            ("RUN_RUNTIME_RAW_LOG", str(self.raw_runtime_log)),
            ("RUN_POLICY_LOG", str(self.policy_server_log)),
            ("RUN_FRONT_VIDEO_FILENAME", self.front_video_filename),
            ("RUN_WRIST_VIDEO_FILENAME", self.wrist_video_filename),
        )
        return "\n".join(shell_assign(name, value) for name, value in values)


@dataclass(frozen=True)
class LingBotBatchAttempt:
    batch_id: str
    task_position: int
    task_count: int
    attempt: int
    attempt_count: int
    sequence: int
    total: int


@dataclass(frozen=True)
class LingBotRunResult:
    run_id: str
    status: str
    evaluation_decision: str = ""

    def shell(self) -> str:
        values = (
            ("RUN_STATUS", self.status),
            ("RUN_EVALUATION_DECISION", self.evaluation_decision),
        )
        return "\n".join(shell_assign(name, value) for name, value in values)


def prepare_lingbot_run(
    config: LingBotConfig,
    task: TaskPrompt,
    *,
    repo_root: Path,
    scene_note: str,
    batch_attempt: LingBotBatchAttempt | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    git_commit: str | None = None,
    git_worktree_dirty: bool | None = None,
) -> LingBotRunPaths:
    root = repo_root.resolve()
    artifacts_root = (output_root or config.recording.output_root).resolve()
    started = now or datetime.now().astimezone()
    note = validate_scene_note(scene_note)
    identity = run_id or recording_run_id(task.task_id, now=started)
    _validate_run_id(identity)
    front_video_filename = lingbot_video_filename(note, task.prompt, now=started)
    wrist_video_filename = lingbot_wrist_video_filename(front_video_filename)
    paths = _run_paths(
        artifacts_root,
        identity,
        front_video_filename=front_video_filename,
        wrist_video_filename=wrist_video_filename,
    )
    video_staging_dir = artifacts_root / f".{identity}.staging"
    occupied = (
        paths.final_dir,
        paths.log_staging_dir,
        video_staging_dir,
    )
    existing = [str(path) for path in occupied if path.exists()]
    if existing:
        raise FileExistsError(f"LingBot run output already exists: {existing}")

    commit, dirty = _git_state(root)
    if git_commit is not None:
        commit = git_commit
    if git_worktree_dirty is not None:
        dirty = git_worktree_dirty
    contract = server_metadata(config)
    context = {
        "schema_version": 1,
        "run_id": identity,
        "started_at": started.isoformat(),
        "scene_note": note,
        "video_filenames": {
            "front": front_video_filename,
            "wrist": wrist_video_filename,
        },
        "task": {
            "id": task.task_id,
            "prompt": task.prompt,
            "distribution": task.distribution,
        },
        "configuration": {
            "deployment": _repo_path(config.path, root),
            "system": _repo_path(config.system.path, root),
            "task_catalog": _repo_path(config.task_catalog.path, root),
            "model": _repo_path(config.policy_server.model.path, root),
            "sha256": {
                "deployment": file_sha256(config.path),
                "system": file_sha256(config.system.path),
                "task_catalog": file_sha256(config.task_catalog.path),
                "model": file_sha256(config.policy_server.model.path),
                "model_contract": file_sha256(config.policy_server.model.contract),
                "model_manifest": file_sha256(config.policy_server.model.manifest.path),
            },
            "model_id": config.policy_server.model.model_id,
            "model_revision": config.policy_server.model.source.revision,
            "server_contract_sha256": contract["contract_sha256"],
        },
        "git": {
            "commit": commit,
            "worktree_dirty": dirty,
        },
    }
    if batch_attempt is not None:
        _validate_batch_attempt(batch_attempt, task_id=task.task_id)
        context["batch"] = {
            "id": batch_attempt.batch_id,
            "task_id": task.task_id,
            "task_position": batch_attempt.task_position,
            "task_count": batch_attempt.task_count,
            "attempt": batch_attempt.attempt,
            "attempt_count": batch_attempt.attempt_count,
            "sequence": batch_attempt.sequence,
            "total": batch_attempt.total,
        }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.log_staging_dir.mkdir()
    atomic_write_json(paths.log_staging_dir / _CONTEXT_NAME, context)
    paths.policy_server_log.touch(exist_ok=False)
    return paths


def record_lingbot_run_outcome(
    output_root: Path,
    run_id: str,
    *,
    kind: str,
    message: str,
) -> None:
    """Record a typed clean-stop outcome for the finalizer."""

    _validate_run_id(run_id)
    if kind not in _OUTCOME_KINDS:
        raise ValueError(f"unsupported LingBot run outcome: {kind!r}")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("LingBot run outcome message must be non-empty")
    paths = _run_paths(output_root.resolve(), run_id)
    context_path = paths.log_staging_dir / _CONTEXT_NAME
    if not context_path.is_file():
        raise FileNotFoundError(f"LingBot run context is missing: {context_path}")
    atomic_write_json(
        paths.log_staging_dir / _OUTCOME_NAME,
        {"kind": kind, "message": message.strip()},
    )


def finalize_lingbot_run(
    output_root: Path,
    run_id: str,
    *,
    exit_code: int,
    now: datetime | None = None,
) -> Path:
    _validate_run_id(run_id)
    if not 0 <= exit_code <= 255:
        raise ValueError(f"run exit code must be in [0, 255], got {exit_code}")
    paths = _run_paths(output_root.resolve(), run_id)
    context_path = paths.log_staging_dir / _CONTEXT_NAME
    if not context_path.is_file():
        if _finalized_run_is_complete(paths):
            return paths.final_dir
        raise FileNotFoundError(f"LingBot run context is missing: {context_path}")
    context = json.loads(context_path.read_text())
    if not isinstance(context, dict) or context.get("run_id") != run_id:
        raise ValueError(f"LingBot run context identity mismatch: {context_path}")
    outcome_path = paths.log_staging_dir / _OUTCOME_NAME
    outcome = _load_outcome(outcome_path) if outcome_path.is_file() else None

    paths.final_dir.mkdir(parents=False, exist_ok=True)
    runtime_log = paths.log_staging_dir / "runtime.log"
    raw_text = (
        paths.raw_runtime_log.read_text(errors="replace")
        if paths.raw_runtime_log.is_file()
        else ""
    )
    runtime_log.write_text(_plain_terminal_log(raw_text))
    if not paths.policy_server_log.exists():
        paths.policy_server_log.touch()

    for source in (runtime_log, paths.policy_server_log):
        destination = paths.final_dir / source.name
        _move_generated_artifact(source, destination)

    metadata_path = paths.final_dir / "metadata.json"
    metadata: dict = {}
    if metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(
                f"LingBot recording metadata must be an object: {metadata_path}"
            )
        metadata = loaded
    ended_at = (now or datetime.now().astimezone()).isoformat()
    if exit_code != 0:
        status = "interrupted" if exit_code in {129, 130, 143} else "failed"
    elif outcome is None:
        status = "completed"
    elif outcome["kind"] in {"ik_target_rejected", "workspace_target_rejected"}:
        status = "safety_stopped"
    else:
        status = "interrupted"
    video_filenames = context.get("video_filenames")
    if not isinstance(video_filenames, dict) or set(video_filenames) != {
        "front",
        "wrist",
    }:
        raise ValueError(
            f"LingBot run video filenames are invalid: {video_filenames!r}"
        )
    for camera, filename in video_filenames.items():
        if (
            not isinstance(filename, str)
            or not filename.endswith(".mp4")
            or Path(filename).name != filename
        ):
            raise ValueError(
                f"LingBot {camera} video filename is invalid: {filename!r}"
            )
    video_paths = {
        camera: paths.final_dir / filename
        for camera, filename in video_filenames.items()
    }
    recording_metadata_path = paths.final_dir / "camera_recording.json"
    recording_metadata = _load_camera_recording_metadata(
        recording_metadata_path,
        expected_videos=video_filenames,
    )
    video_staging_dir = paths.output_root / f".{run_id}.staging"
    run_metadata = {
        key: value for key, value in context.items() if key != "schema_version"
    } | {
        "ended_at": ended_at,
        "exit_code": exit_code,
        "status": status,
    }
    if outcome is not None:
        run_metadata["termination"] = outcome
    metadata.update(
        {
            "schema_version": 3,
            "run": run_metadata,
            "artifacts": {
                "videos": {
                    camera: path.name if path.is_file() else None
                    for camera, path in video_paths.items()
                },
                "camera_timeline": (
                    recording_metadata["timeline"]
                    if recording_metadata is not None
                    else None
                ),
                "camera_recording": (
                    recording_metadata_path.name
                    if recording_metadata is not None
                    else None
                ),
                "runtime_log": "runtime.log",
                "policy_server_log": "policy_server.log",
                "incomplete_video_staging": video_staging_dir.exists(),
            },
        }
    )
    if recording_metadata is not None:
        metadata.update(
            {
                "frames": recording_metadata["frames"],
                "fps": recording_metadata["fps"],
                "elapsed_s": recording_metadata["elapsed_s"],
            }
        )
    atomic_write_json(metadata_path, metadata)

    context_path.unlink()
    if outcome_path.exists():
        outcome_path.unlink()
    if paths.raw_runtime_log.exists():
        paths.raw_runtime_log.unlink()
    paths.log_staging_dir.rmdir()
    return paths.final_dir


def inspect_lingbot_run(output_root: Path, run_id: str) -> LingBotRunResult:
    """Load the finalized status and optional operator evaluation decision."""

    _validate_run_id(run_id)
    paths = _run_paths(output_root.resolve(), run_id)
    metadata = _load_finalized_metadata(paths)
    run = metadata["run"]
    status = run.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"LingBot run status is invalid: {paths.final_dir}")
    decision = _evaluation_decision(run, paths.final_dir)
    return LingBotRunResult(
        run_id=run_id,
        status=status,
        evaluation_decision=decision or "",
    )


def record_lingbot_evaluation_decision(
    output_root: Path,
    run_id: str,
    *,
    decision: str,
    now: datetime | None = None,
) -> Path:
    """Persist whether an operator counts or discards a safety-stopped run."""

    _validate_run_id(run_id)
    if decision not in _EVALUATION_DECISIONS:
        raise ValueError(f"unsupported LingBot evaluation decision: {decision!r}")
    paths = _run_paths(output_root.resolve(), run_id)
    metadata = _load_finalized_metadata(paths)
    run = metadata["run"]
    status = run.get("status")
    if status != "safety_stopped":
        raise ValueError(
            "evaluation decisions apply only to safety-stopped runs, "
            f"got status {status!r}"
        )
    existing = _evaluation_decision(run, paths.final_dir)
    if existing is not None and existing != decision:
        raise ValueError(
            f"LingBot evaluation decision is already {existing!r}: {paths.final_dir}"
        )
    metadata_path = paths.final_dir / "metadata.json"
    if existing == decision:
        return metadata_path
    run["evaluation"] = {
        "decision": decision,
        "decided_at": (now or datetime.now().astimezone()).isoformat(),
    }
    metadata["run"] = run
    atomic_write_json(metadata_path, metadata)
    return metadata_path


def _run_paths(
    output_root: Path,
    run_id: str,
    *,
    front_video_filename: str = "",
    wrist_video_filename: str = "",
) -> LingBotRunPaths:
    staging = output_root / f".{run_id}.logs"
    return LingBotRunPaths(
        run_id=run_id,
        output_root=output_root,
        final_dir=output_root / run_id,
        log_staging_dir=staging,
        raw_runtime_log=staging / _RAW_RUNTIME_LOG_NAME,
        policy_server_log=staging / "policy_server.log",
        front_video_filename=front_video_filename,
        wrist_video_filename=wrist_video_filename,
    )


def lingbot_video_filename(
    scene_note: str,
    prompt: str,
    *,
    now: datetime | None = None,
) -> str:
    note = _filename_component(validate_scene_note(scene_note), max_bytes=60)
    prompt_component = _filename_component(prompt, max_bytes=140)
    date = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    return f"{note}__{prompt_component}__{date}__front.mp4"


def lingbot_wrist_video_filename(front_video_filename: str) -> str:
    if (
        not front_video_filename.endswith("__front.mp4")
        or Path(front_video_filename).name != front_video_filename
    ):
        raise ValueError(
            f"invalid LingBot front video filename: {front_video_filename!r}"
        )
    filename = f"{front_video_filename[: -len('__front.mp4')]}__wrist.mp4"
    if len(filename.encode("utf-8")) > 240:
        raise ValueError(f"LingBot wrist video filename is too long: {filename!r}")
    return filename


def _filename_component(value: str, *, max_bytes: int) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    output: list[str] = []
    separator_pending = False
    for character in normalized:
        if character.isalnum():
            if separator_pending and output:
                output.append("_")
            output.append(character)
            separator_pending = False
        else:
            separator_pending = True
    component = "".join(output).strip("_")
    encoded = bytearray()
    for character in component:
        character_bytes = character.encode("utf-8")
        if len(encoded) + len(character_bytes) > max_bytes:
            break
        encoded.extend(character_bytes)
    return encoded.decode("utf-8").rstrip("_") or "note"


def _validate_batch_attempt(value: LingBotBatchAttempt, *, task_id: str) -> None:
    slot = EvaluationSlot(
        plan_id=value.batch_id,
        task_id=task_id,
        task_position=value.task_position,
        task_count=value.task_count,
        attempt=value.attempt,
        attempt_count=value.attempt_count,
    )
    if value.total != slot.total or value.sequence != slot.sequence:
        raise ValueError(
            "inconsistent LingBot batch attempt indices: "
            f"sequence={value.sequence}/{value.total}, expected "
            f"{slot.sequence}/{slot.total}"
        )


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or run_id.startswith(".")
        or any(
            not (character.isalnum() or character in {"-", "_", "."})
            for character in run_id
        )
    ):
        raise ValueError(f"invalid LingBot run id: {run_id!r}")


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _repo_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _plain_terminal_log(value: str) -> str:
    plain = strip_protocol_events(
        _ANSI_ESCAPE.sub("", value).replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [line.rstrip() for line in plain.splitlines()]
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def _move_generated_artifact(source: Path, destination: Path) -> None:
    if not source.exists():
        if destination.is_file():
            return
        raise FileNotFoundError(f"LingBot run artifact is missing: {source}")
    if destination.exists():
        if not destination.is_file() or source.read_bytes() != destination.read_bytes():
            raise FileExistsError(
                f"LingBot run artifact conflicts with existing output: {destination}"
            )
        source.unlink()
        return
    os.replace(source, destination)


def _load_camera_recording_metadata(
    path: Path,
    *,
    expected_videos: dict[str, str],
) -> dict | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text())
    expected_keys = {
        "schema_version",
        "videos",
        "timeline",
        "fps",
        "frames",
        "elapsed_s",
        "started_at",
        "ended_at",
        "max_source_age_s",
        "max_pair_skew_s",
    }
    if not isinstance(loaded, dict) or set(loaded) != expected_keys:
        raise ValueError(f"LingBot camera recording metadata is invalid: {path}")
    if loaded["schema_version"] != 1:
        raise ValueError(f"LingBot camera recording schema is unsupported: {path}")
    videos = loaded["videos"]
    if not isinstance(videos, dict) or set(videos) != {"front", "wrist"}:
        raise ValueError(f"LingBot camera recording videos are invalid: {path}")
    for camera, expected_filename in expected_videos.items():
        entry = videos.get(camera)
        if not isinstance(entry, dict) or entry.get("file") != expected_filename:
            raise ValueError(
                f"LingBot {camera} recording does not match run context: {path}"
            )
        if set(entry) != {"file", "source", "width", "height"}:
            raise ValueError(f"LingBot {camera} recording is invalid: {path}")
        if (
            not isinstance(entry["source"], str)
            or not entry["source"]
            or not isinstance(entry["width"], int)
            or isinstance(entry["width"], bool)
            or entry["width"] <= 0
            or not isinstance(entry["height"], int)
            or isinstance(entry["height"], bool)
            or entry["height"] <= 0
        ):
            raise ValueError(f"LingBot {camera} recording is invalid: {path}")
        video_path = path.parent / expected_filename
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise ValueError(
                f"LingBot {camera} recording video is missing: {video_path}"
            )
    timeline = loaded["timeline"]
    if (
        not isinstance(timeline, str)
        or Path(timeline).name != timeline
        or not (path.parent / timeline).is_file()
    ):
        raise ValueError(f"LingBot camera recording timeline is invalid: {path}")
    frames = loaded["frames"]
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise ValueError(f"LingBot camera recording frame count is invalid: {path}")
    for key in (
        "fps",
        "elapsed_s",
        "max_source_age_s",
        "max_pair_skew_s",
    ):
        value = loaded[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"LingBot camera recording {key} is invalid: {path}")
    if loaded["fps"] <= 0 or loaded["max_source_age_s"] <= 0:
        raise ValueError(f"LingBot camera recording timing is invalid: {path}")
    if not all(
        isinstance(loaded[key], str) and loaded[key]
        for key in ("started_at", "ended_at")
    ):
        raise ValueError(f"LingBot camera recording timestamps are invalid: {path}")
    return loaded


def _finalized_run_is_complete(paths: LingBotRunPaths) -> bool:
    metadata_path = paths.final_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and isinstance(metadata.get("run"), dict)
        and metadata["run"].get("run_id") == paths.run_id
        and (paths.final_dir / "runtime.log").is_file()
        and (paths.final_dir / "policy_server.log").is_file()
    )


def _load_finalized_metadata(paths: LingBotRunPaths) -> dict:
    metadata_path = paths.final_dir / "metadata.json"
    if not _finalized_run_is_complete(paths):
        raise FileNotFoundError(
            f"LingBot finalized run is incomplete: {paths.final_dir}"
        )
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("run"), dict):
        raise ValueError(f"LingBot run metadata is invalid: {metadata_path}")
    return metadata


def _evaluation_decision(run: dict, run_dir: Path) -> str | None:
    evaluation = run.get("evaluation")
    if evaluation is None:
        return None
    if not isinstance(evaluation, dict):
        raise ValueError(f"LingBot evaluation decision is invalid: {run_dir}")
    decision = evaluation.get("decision")
    decided_at = evaluation.get("decided_at")
    if (
        set(evaluation) != {"decision", "decided_at"}
        or decision not in _EVALUATION_DECISIONS
        or not isinstance(decided_at, str)
        or not decided_at
    ):
        raise ValueError(f"LingBot evaluation decision is invalid: {run_dir}")
    return decision


def _load_outcome(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"kind", "message"}:
        raise ValueError(f"LingBot run outcome is invalid: {path}")
    kind = value.get("kind")
    message = value.get("message")
    if (
        kind not in _OUTCOME_KINDS
        or not isinstance(message, str)
        or not message.strip()
    ):
        raise ValueError(f"LingBot run outcome is invalid: {path}")
    return {"kind": kind, "message": message.strip()}


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, default=ROOT)
    prepare.add_argument("--config", type=Path, default=default_config_path(ROOT))
    prepare.add_argument("--model")
    prepare.add_argument("--task-id", required=True)
    prepare.add_argument("--scene-note", required=True)
    prepare.add_argument("--batch-id")
    prepare.add_argument("--task-position", type=int)
    prepare.add_argument("--task-count", type=int)
    prepare.add_argument("--attempt", type=int)
    prepare.add_argument("--attempt-count", type=int)
    prepare.add_argument("--sequence", type=int)
    prepare.add_argument("--total", type=int)
    prepare.add_argument("--shell", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--output-root", type=Path, required=True)
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--shell", action="store_true")
    decide = subparsers.add_parser("decide")
    decide.add_argument("--output-root", type=Path, required=True)
    decide.add_argument("--run-id", required=True)
    decide.add_argument(
        "--decision", choices=sorted(_EVALUATION_DECISIONS), required=True
    )
    args = parser.parse_args()

    if args.command == "prepare":
        config = load_lingbot_config(
            args.config,
            repo_root=args.repo_root,
            model_selector=args.model,
        )
        batch_values = (
            args.batch_id,
            args.task_position,
            args.task_count,
            args.attempt,
            args.attempt_count,
            args.sequence,
            args.total,
        )
        if any(value is not None for value in batch_values) and not all(
            value is not None for value in batch_values
        ):
            parser.error("all batch attempt fields must be provided together")
        batch_attempt = (
            LingBotBatchAttempt(*batch_values)
            if all(value is not None for value in batch_values)
            else None
        )
        paths = prepare_lingbot_run(
            config,
            config.task_catalog.task(args.task_id),
            repo_root=args.repo_root,
            scene_note=args.scene_note,
            batch_attempt=batch_attempt,
        )
        print(paths.shell() if args.shell else paths.run_id)
        return 0
    if args.command == "finalize":
        result = finalize_lingbot_run(
            args.output_root,
            args.run_id,
            exit_code=args.exit_code,
        )
        print(result)
        return 0
    if args.command == "inspect":
        result = inspect_lingbot_run(args.output_root, args.run_id)
        print(result.shell() if args.shell else result)
        return 0
    result = record_lingbot_evaluation_decision(
        args.output_root,
        args.run_id,
        decision=args.decision,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
