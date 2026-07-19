"""Report and export valid LingBot batch evaluations."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import unicodedata
from datetime import datetime
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.batch_config import (
    DEFAULT_BATCH_CONFIG,
    LingBotBatchConfig,
    load_lingbot_batch_config,
)
from galaxea_a1_runtime.apps.lingbot.batch_progress import (
    LingBotBatchProgress,
    LingBotValidBatchRun,
    inspect_lingbot_batch_progress,
)
from galaxea_a1_runtime.apps.lingbot.operator_input import validate_scene_note
from galaxea_a1_runtime.console import ArgumentParser


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPORT_ROOT = ROOT / "outputs/exports/lingbot"


def render_lingbot_batch_report(
    config: LingBotBatchConfig,
    *,
    scene_note: str,
    recording_root: Path | None = None,
) -> str:
    note = validate_scene_note(scene_note)
    progress = inspect_lingbot_batch_progress(
        config,
        scene_note=note,
        output_root=recording_root,
    )
    by_sequence = _runs_by_sequence(progress)
    lines = [
        f"Batch: {config.batch_id}",
        f"Scene note: {note}",
        f"Valid slots: {progress.completed_count}/{progress.total}",
        f"Pending slots: {progress.pending_count}",
        f"Duplicate valid slots: {_csv(progress.duplicate_sequences) or 'none'}",
        "",
    ]
    for sequence in range(1, config.total_attempts + 1):
        task_position = (sequence - 1) // config.attempts_per_prompt
        attempt = (sequence - 1) % config.attempts_per_prompt + 1
        task_id = config.task_ids[task_position]
        runs = by_sequence.get(sequence, ())
        if not runs:
            result = "PENDING"
        elif len(runs) > 1:
            result = "DUPLICATE_VALID " + ",".join(run.run_id for run in runs)
        else:
            run = runs[0]
            result = (
                f"VALID status={run.status} decision={run.evaluation_decision} "
                f"run={run.run_id}"
            )
        lines.append(
            f"{sequence:02d}/{config.total_attempts}: task={task_id} "
            f"attempt={attempt}/{config.attempts_per_prompt} {result}"
        )
    return "\n".join(lines) + "\n"


def export_valid_lingbot_batch(
    config: LingBotBatchConfig,
    *,
    scene_note: str,
    recording_root: Path | None = None,
    export_root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Atomically create one tar containing exactly one valid run per slot."""

    note = validate_scene_note(scene_note)
    progress = inspect_lingbot_batch_progress(
        config,
        scene_note=note,
        output_root=recording_root,
    )
    if progress.duplicate_sequences:
        raise ValueError(
            "LingBot batch has duplicate valid slots: "
            f"{_csv(progress.duplicate_sequences)}"
        )
    pending = tuple(
        sequence
        for sequence in range(1, progress.total + 1)
        if sequence not in set(progress.completed_sequences)
    )
    if pending:
        raise ValueError(f"LingBot batch is incomplete; pending slots: {_csv(pending)}")

    created = now or datetime.now().astimezone()
    destination_root = (export_root or DEFAULT_EXPORT_ROOT).resolve()
    filename = (
        f"{config.batch_id}__{_filename_component(note)}__"
        f"{created.strftime('%Y%m%d_%H%M%S')}.tar"
    )
    destination = destination_root / filename
    temporary = destination_root / f".{filename}.tmp.{os.getpid()}"
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"LingBot batch export already exists: {destination}")

    by_sequence = _runs_by_sequence(progress)
    manifest_runs = []
    archive_files: list[tuple[Path, str]] = []
    for sequence in range(1, config.total_attempts + 1):
        run = by_sequence[sequence][0]
        metadata = json.loads((run.run_dir / "metadata.json").read_text())
        artifacts = metadata["artifacts"]
        names = (
            "metadata.json",
            artifacts["runtime_log"],
            artifacts["policy_server_log"],
            artifacts["video"],
        )
        archive_dir = f"runs/{sequence:02d}_{run.run_id}"
        files = []
        for name in names:
            source = run.run_dir / name
            _require_regular_file(source)
            archive_path = f"{archive_dir}/{name}"
            archive_files.append((source, archive_path))
            files.append(
                {
                    "path": archive_path,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        task = config.deployment.task_catalog.task(
            config.task_ids[run.task_position - 1]
        )
        manifest_runs.append(
            {
                "sequence": sequence,
                "task_position": run.task_position,
                "attempt": run.attempt,
                "run_id": run.run_id,
                "status": run.status,
                "evaluation_decision": run.evaluation_decision,
                "legacy_safety_stop": run.legacy_safety_stop,
                "task": {
                    "id": task.task_id,
                    "prompt": task.prompt,
                    "distribution": task.distribution,
                },
                "files": files,
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": created.isoformat(),
        "batch_id": config.batch_id,
        "scene_note": note,
        "batch_config": _portable_path(config.path),
        "total_slots": config.total_attempts,
        "runs": manifest_runs,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    try:
        with tarfile.open(temporary, mode="x") as archive:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(created.timestamp())
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for source, archive_path in archive_files:
                archive.add(source, arcname=archive_path, recursive=False)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _runs_by_sequence(
    progress: LingBotBatchProgress,
) -> dict[int, tuple[LingBotValidBatchRun, ...]]:
    grouped: dict[int, list[LingBotValidBatchRun]] = {}
    for run in progress.valid_runs:
        grouped.setdefault(run.sequence, []).append(run)
    return {sequence: tuple(runs) for sequence, runs in grouped.items()}


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FileNotFoundError(f"LingBot export artifact is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"LingBot export artifact is not a regular file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filename_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    component = "_".join(normalized.split())
    component = "".join(
        character
        for character in component
        if character.isalnum() or character in {"-", "_"}
    )
    encoded = bytearray()
    for character in component:
        character_bytes = character.encode("utf-8")
        if len(encoded) + len(character_bytes) > 60:
            break
        encoded.extend(character_bytes)
    return encoded.decode("utf-8").rstrip("_") or "scene"


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "export"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=ROOT)
        command.add_argument("--model")
        command.add_argument("--scene-note", required=True)
        command.add_argument("--recording-root", type=Path)
        if name == "export":
            command.add_argument("--export-root", type=Path)
        command.add_argument(
            "config", type=Path, nargs="?", default=DEFAULT_BATCH_CONFIG
        )
    args = parser.parse_args()
    config = load_lingbot_batch_config(
        args.config,
        repo_root=args.repo_root,
        model_selector=args.model,
    )
    if args.command == "report":
        print(
            render_lingbot_batch_report(
                config,
                scene_note=args.scene_note,
                recording_root=args.recording_root,
            ),
            end="",
        )
        return 0
    result = export_valid_lingbot_batch(
        config,
        scene_note=args.scene_note,
        recording_root=args.recording_root,
        export_root=args.export_root,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
