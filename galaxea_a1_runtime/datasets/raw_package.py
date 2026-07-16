"""Build a reviewable Hugging Face raw-data package from copied raw snapshots.

The source trees are treated as immutable. Every source task is copied into an
ephemeral sibling snapshot before an archive is created, and each archive is
extracted and compared with that snapshot before the final package is
atomically installed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.collection.episode_output import validate_staged_episode
from galaxea_a1_runtime.collection.schema import (
    TELEOP_RAW_SCHEMA_VERSION,
    validate_experiment_name,
)
from galaxea_a1_runtime.configuration.base import (
    boolean,
    integer,
    load_toml,
    repo_path,
    require_exact_keys,
    required_table,
    string,
)
from galaxea_a1_runtime.console import ArgumentParser, failure, info, step, success
from galaxea_a1_runtime.filesystem import atomic_output_directory

RAW_PACKAGE_FORMAT = "galaxea_a1_teleop_raw_v3_episode_archives_v1"
ARCHIVE_FORMAT = "tar.zst"


@dataclass(frozen=True)
class RawPackageConfig:
    path: Path
    repo_root: Path
    repo_id: str
    output_root: Path
    readme_scene_image: Path
    source_roots: tuple[Path, ...]
    archive_format: str
    zstd_level: int
    overwrite: bool


@dataclass(frozen=True)
class TreeSummary:
    file_count: int
    total_bytes: int
    sha256: str


@dataclass(frozen=True)
class RawContract:
    fps_target: float
    state_mode: str
    action_mode: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    cameras: tuple[dict[str, Any], ...]
    config_path: str

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "action_mode": self.action_mode,
                "action_names": self.action_names,
                "cameras": self.cameras,
                "config_path": self.config_path,
                "fps_target": self.fps_target,
                "state_mode": self.state_mode,
                "state_names": self.state_names,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    name: str
    index: int
    frame_count: int
    summary: TreeSummary


@dataclass(frozen=True)
class RawTask:
    root: Path
    slug: str
    prompt: str
    contract: RawContract
    episodes: tuple[RawEpisode, ...]
    summary: TreeSummary
    task_txt_sha256: str

    @property
    def total_frames(self) -> int:
        return sum(episode.frame_count for episode in self.episodes)


@dataclass(frozen=True)
class RawPackageResult:
    output_root: Path
    readme_path: Path
    manifest_path: Path
    total_tasks: int
    total_episodes: int
    total_frames: int
    archive_bytes: int


def load_raw_package_config(
    path: Path, *, repo_root: Path | None = None
) -> RawPackageConfig:
    config_path, root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(data, required={"raw_package"}, label="dataset config")
    package = required_table(data, "raw_package")
    require_exact_keys(
        package,
        required={
            "repo_id",
            "output_root",
            "readme_scene_image",
            "source_roots",
            "archive_format",
            "zstd_level",
            "overwrite",
        },
        label="raw_package",
    )

    repo_id = string(package, "repo_id")
    if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise ValueError("raw_package.repo_id must be a namespaced Hub repo id")

    archive_format = string(package, "archive_format")
    if archive_format != ARCHIVE_FORMAT:
        raise ValueError(
            f"raw_package.archive_format must be {ARCHIVE_FORMAT!r}, "
            f"got {archive_format!r}"
        )

    zstd_level = integer(package, "zstd_level")
    if not 1 <= zstd_level <= 19:
        raise ValueError("raw_package.zstd_level must be between 1 and 19")

    source_values = package.get("source_roots")
    if (
        not isinstance(source_values, list)
        or not source_values
        or not all(isinstance(item, str) and item for item in source_values)
    ):
        raise ValueError("raw_package.source_roots must be a non-empty string list")
    source_roots = tuple(repo_path(root, value) for value in source_values)
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("raw_package.source_roots contains duplicates")

    raw_root = (root / "data" / "raw").resolve()
    for source_root in source_roots:
        if not source_root.is_relative_to(raw_root):
            raise ValueError(
                f"raw package source must be below {raw_root}: {source_root}"
            )
        validate_experiment_name(source_root.name)

    output_root = repo_path(root, string(package, "output_root"))
    export_root = (root / "data" / "exports").resolve()
    if not output_root.is_relative_to(export_root) or output_root == export_root:
        raise ValueError(
            f"raw_package.output_root must be below {export_root}: {output_root}"
        )

    readme_scene_image = repo_path(root, string(package, "readme_scene_image"))
    assets_root = (root / "assets").resolve()
    if not readme_scene_image.is_relative_to(assets_root):
        raise ValueError(
            f"raw_package.readme_scene_image must be below {assets_root}: "
            f"{readme_scene_image}"
        )

    return RawPackageConfig(
        path=config_path,
        repo_root=root,
        repo_id=repo_id,
        output_root=output_root,
        readme_scene_image=readme_scene_image,
        source_roots=source_roots,
        archive_format=archive_format,
        zstd_level=zstd_level,
        overwrite=boolean(package, "overwrite"),
    )


def build_raw_package(
    config: RawPackageConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> RawPackageResult:
    """Build and atomically install one raw package without mutating sources."""

    if (
        not config.readme_scene_image.is_file()
        or config.readme_scene_image.is_symlink()
    ):
        raise ValueError(
            "raw package README scene image must be a regular file: "
            f"{config.readme_scene_image}"
        )
    _require_archive_tools()
    _reject_stale_staging(config.output_root)
    report = progress or (lambda _message: None)

    report("Validating and hashing immutable raw sources")
    source_tasks = tuple(_inspect_task(root) for root in config.source_roots)
    contract = _shared_contract(source_tasks)

    with atomic_output_directory(
        config.output_root, overwrite=config.overwrite
    ) as staging:
        snapshot_root = staging / ".source-snapshot"
        snapshot_root.mkdir()

        report("Copying raw sources into the isolated staging snapshot")
        for task in source_tasks:
            destination = snapshot_root / task.slug
            shutil.copytree(task.root, destination, copy_function=shutil.copy2)

        report("Hashing the copied snapshot and comparing it with the sources")
        snapshot_tasks = tuple(
            _inspect_task(snapshot_root / task.slug) for task in source_tasks
        )
        _assert_task_copies_match(source_tasks, snapshot_tasks)
        _shared_contract(snapshot_tasks)

        archive_records: list[dict[str, Any]] = []
        total_episodes = sum(len(task.episodes) for task in snapshot_tasks)
        completed = 0
        verify_root = staging / ".archive-verification"
        for task in snapshot_tasks:
            task_output = staging / "tasks" / task.slug
            archive_dir = task_output / "episodes"
            archive_dir.mkdir(parents=True)
            shutil.copy2(task.root / "task.txt", task_output / "task.txt")
            for episode in task.episodes:
                completed += 1
                report(
                    f"Packaging and verifying episode {completed}/{total_episodes}: "
                    f"{task.slug}/{episode.name}"
                )
                archive_path = archive_dir / f"{episode.name}.tar.zst"
                _create_episode_archive(
                    episode.path,
                    archive_path,
                    zstd_level=config.zstd_level,
                )
                _verify_episode_archive(
                    archive_path,
                    episode_name=episode.name,
                    expected=episode.summary,
                    verify_root=verify_root,
                )
                archive_records.append(
                    {
                        "archive": archive_path.relative_to(staging).as_posix(),
                        "archive_bytes": archive_path.stat().st_size,
                        "archive_sha256": _file_sha256(archive_path),
                        "episode": episode.name,
                        "episode_index": episode.index,
                        "fps": task.contract.fps_target,
                        "frame_count": episode.frame_count,
                        "source_bytes": episode.summary.total_bytes,
                        "source_file_count": episode.summary.file_count,
                        "source_tree_sha256": episode.summary.sha256,
                        "task": task.prompt,
                        "task_slug": task.slug,
                    }
                )

        shutil.rmtree(snapshot_root)
        if verify_root.exists():
            shutil.rmtree(verify_root)

        report("Re-hashing raw sources to confirm they did not change during build")
        final_source_tasks = tuple(_inspect_task(root) for root in config.source_roots)
        _assert_task_copies_match(source_tasks, final_source_tasks)

        manifest = _build_manifest(
            config=config,
            tasks=source_tasks,
            contract=contract,
            archive_records=archive_records,
        )
        _write_json(staging / "manifest.json", manifest)
        _write_jsonl(staging / "episodes.jsonl", archive_records)
        _write_checksums(staging / "checksums.sha256", archive_records)
        readme_assets = staging / "assets"
        readme_assets.mkdir()
        shutil.copy2(
            config.readme_scene_image,
            readme_assets / config.readme_scene_image.name,
        )
        (staging / "README.md").write_text(
            _render_readme(
                config=config,
                tasks=source_tasks,
                contract=contract,
                archive_records=archive_records,
            )
        )

    return RawPackageResult(
        output_root=config.output_root,
        readme_path=config.output_root / "README.md",
        manifest_path=config.output_root / "manifest.json",
        total_tasks=len(source_tasks),
        total_episodes=len(archive_records),
        total_frames=sum(task.total_frames for task in source_tasks),
        archive_bytes=sum(record["archive_bytes"] for record in archive_records),
    )


def _inspect_task(root: Path) -> RawTask:
    if not root.is_dir():
        raise ValueError(f"raw task root does not exist: {root}")
    _reject_symlinks(root)
    validate_experiment_name(root.name)

    children = sorted(root.iterdir(), key=lambda path: path.name)
    unexpected = [
        child.name
        for child in children
        if child.name != "task.txt"
        and not (child.is_dir() and child.name.startswith("episode_"))
    ]
    if unexpected:
        raise ValueError(f"unexpected entries in raw task {root}: {unexpected[:5]}")

    task_path = root / "task.txt"
    if not task_path.is_file():
        raise ValueError(f"raw task is missing task.txt: {root}")
    prompt = _clean_text(task_path.read_text())
    if not prompt:
        raise ValueError(f"raw task prompt is empty: {task_path}")

    episode_dirs = [
        child
        for child in children
        if child.is_dir() and child.name.startswith("episode_")
    ]
    if not episode_dirs:
        raise ValueError(f"raw task has no episodes: {root}")

    episodes: list[RawEpisode] = []
    expected_contract: RawContract | None = None
    seen_indices: set[int] = set()
    for episode_dir in episode_dirs:
        metadata_path = episode_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid raw metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"raw metadata must be an object: {metadata_path}")
        if metadata.get("schema_version") != TELEOP_RAW_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported raw schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}"
            )
        if metadata.get("experiment") != root.name:
            raise ValueError(
                f"metadata experiment does not match task directory: {metadata_path}"
            )
        if _clean_text(metadata.get("task")) != prompt:
            raise ValueError(f"metadata task does not match task.txt: {metadata_path}")

        episode_index = _nonnegative_integer(metadata, "episode_index", metadata_path)
        expected_name = f"episode_{episode_index:03d}"
        if episode_dir.name != expected_name and not episode_dir.name.startswith(
            expected_name + "_"
        ):
            raise ValueError(
                f"episode directory {episode_dir.name!r} does not match "
                f"metadata index {episode_index}"
            )
        if episode_index in seen_indices:
            raise ValueError(f"duplicate episode index {episode_index} in {root}")
        seen_indices.add(episode_index)

        frame_count = _positive_integer(metadata, "frame_count", metadata_path)
        cameras = _camera_list(metadata, metadata_path)
        depth_enabled = any(camera.get("modality") == "depth" for camera in cameras)
        validate_staged_episode(
            episode_dir,
            frame_count=frame_count,
            depth_enabled=depth_enabled,
        )
        contract = _contract_from_metadata(metadata, cameras, metadata_path)
        if expected_contract is None:
            expected_contract = contract
        elif contract.canonical_json() != expected_contract.canonical_json():
            raise ValueError(f"raw episode contract drift in {metadata_path}")
        episodes.append(
            RawEpisode(
                path=episode_dir,
                name=episode_dir.name,
                index=episode_index,
                frame_count=frame_count,
                summary=_summarize_tree(episode_dir),
            )
        )

    assert expected_contract is not None
    return RawTask(
        root=root,
        slug=root.name,
        prompt=prompt,
        contract=expected_contract,
        episodes=tuple(episodes),
        summary=_summarize_tree(root),
        task_txt_sha256=_file_sha256(task_path),
    )


def _contract_from_metadata(
    metadata: dict[str, Any],
    cameras: tuple[dict[str, Any], ...],
    metadata_path: Path,
) -> RawContract:
    fps = metadata.get("fps_target")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"metadata fps_target must be positive: {metadata_path}")
    return RawContract(
        fps_target=float(fps),
        state_mode=_required_metadata_string(metadata, "state_mode", metadata_path),
        action_mode=_required_metadata_string(metadata, "action_mode", metadata_path),
        state_names=_metadata_names(metadata, "state_names", metadata_path),
        action_names=_metadata_names(metadata, "action_names", metadata_path),
        cameras=cameras,
        config_path=_required_metadata_string(metadata, "config_path", metadata_path),
    )


def _camera_list(
    metadata: dict[str, Any], metadata_path: Path
) -> tuple[dict[str, Any], ...]:
    value = metadata.get("cameras")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(camera, dict) for camera in value)
    ):
        raise ValueError(
            f"metadata cameras must be a non-empty object list: {metadata_path}"
        )
    names = [camera.get("name") for camera in value]
    directories = [camera.get("directory") for camera in value]
    if any(not isinstance(item, str) or not item for item in (*names, *directories)):
        raise ValueError(f"camera name/directory must be non-empty: {metadata_path}")
    if len(set(names)) != len(names) or len(set(directories)) != len(directories):
        raise ValueError(f"camera names/directories must be unique: {metadata_path}")
    return tuple(value)


def _shared_contract(tasks: tuple[RawTask, ...]) -> RawContract:
    if not tasks:
        raise ValueError("raw package requires at least one task")
    expected = tasks[0].contract
    for task in tasks[1:]:
        if task.contract.canonical_json() != expected.canonical_json():
            raise ValueError(
                f"raw task contract drift between {tasks[0].slug} and {task.slug}"
            )
    return expected


def _assert_task_copies_match(
    expected: tuple[RawTask, ...], actual: tuple[RawTask, ...]
) -> None:
    if len(expected) != len(actual):
        raise RuntimeError("raw task count changed while copying")
    for source, copy in zip(expected, actual, strict=True):
        if source.slug != copy.slug or source.summary != copy.summary:
            raise RuntimeError(
                f"raw tree changed or copied incorrectly: {source.slug} "
                f"expected={source.summary}, actual={copy.summary}"
            )


def _summarize_tree(root: Path) -> TreeSummary:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError(f"symlinks are not supported in raw data: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sha256 = _file_sha256(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(sha256.encode())
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise ValueError(f"raw tree contains no files: {root}")
    return TreeSummary(
        file_count=file_count,
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_episode_archive(
    episode_dir: Path, archive_path: Path, *, zstd_level: int
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tar_command = [
        "tar",
        "--sort=name",
        "--format=ustar",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--mode=u+rwX,go+rX,go-w",
        "-C",
        str(episode_dir.parent),
        "-cf",
        "-",
        episode_dir.name,
    ]
    zstd_command = [
        "zstd",
        f"-{zstd_level}",
        "--threads=1",
        "--quiet",
        "--force",
        "-o",
        str(archive_path),
    ]
    tar_process = subprocess.Popen(
        tar_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert tar_process.stdout is not None
    zstd_result = subprocess.run(
        zstd_command,
        stdin=tar_process.stdout,
        capture_output=True,
        check=False,
    )
    tar_process.stdout.close()
    assert tar_process.stderr is not None
    tar_stderr = tar_process.stderr.read()
    tar_returncode = tar_process.wait()
    if tar_returncode != 0 or zstd_result.returncode != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "episode archive failed: "
            f"tar={tar_returncode} {tar_stderr.decode(errors='replace').strip()!r}, "
            f"zstd={zstd_result.returncode} "
            f"{zstd_result.stderr.decode(errors='replace').strip()!r}"
        )


def _verify_episode_archive(
    archive_path: Path,
    *,
    episode_name: str,
    expected: TreeSummary,
    verify_root: Path,
) -> None:
    if verify_root.exists():
        shutil.rmtree(verify_root)
    verify_root.mkdir()
    result = subprocess.run(
        [
            "tar",
            "--use-compress-program=unzstd",
            "--no-same-owner",
            "-xf",
            str(archive_path),
            "-C",
            str(verify_root),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"archive verification extraction failed for {archive_path}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    extracted = verify_root / episode_name
    actual = _summarize_tree(extracted)
    if actual != expected:
        raise RuntimeError(
            f"archive content mismatch for {archive_path}: "
            f"expected={expected}, actual={actual}"
        )
    shutil.rmtree(verify_root)


def _require_archive_tools() -> None:
    missing = [name for name in ("tar", "zstd", "unzstd") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"raw packaging requires missing tools: {missing}")


def _reject_stale_staging(output_root: Path) -> None:
    if not output_root.parent.exists():
        return
    stale = sorted(
        path
        for path in output_root.parent.iterdir()
        if path.name.startswith(f".{output_root.name}.staging-")
    )
    if stale:
        names = [path.name for path in stale]
        raise RuntimeError(
            "raw package crash leftovers require operator inspection before retry: "
            f"{names}"
        )


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"raw task root must not be a symlink: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not supported in raw data: {path}")


def _build_manifest(
    *,
    config: RawPackageConfig,
    tasks: tuple[RawTask, ...],
    contract: RawContract,
    archive_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "archive_format": config.archive_format,
        "archive_member_root": "episode directory",
        "archive_sha256_scope": "compressed tar.zst bytes",
        "archive_total_bytes": sum(item["archive_bytes"] for item in archive_records),
        "compression": {"codec": "zstd", "level": config.zstd_level, "threads": 1},
        "format": RAW_PACKAGE_FORMAT,
        "raw_contract": {
            "action_mode": contract.action_mode,
            "action_names": list(contract.action_names),
            "cameras": list(contract.cameras),
            "collection_config": contract.config_path,
            "fps": contract.fps_target,
            "state_mode": contract.state_mode,
            "state_names": list(contract.state_names),
        },
        "raw_schema_version": TELEOP_RAW_SCHEMA_VERSION,
        "readme_scene_image": {
            "path": f"assets/{config.readme_scene_image.name}",
            "sha256": _file_sha256(config.readme_scene_image),
        },
        "repo_id": config.repo_id,
        "source_tree_sha256_algorithm": (
            "sha256 of sorted UTF-8 relative_path\\0decimal_size\\0file_sha256\\n records"
        ),
        "tasks": [
            {
                "episodes": len(task.episodes),
                "frames": task.total_frames,
                "prompt": task.prompt,
                "slug": task.slug,
                "source_bytes": task.summary.total_bytes,
                "source_file_count": task.summary.file_count,
                "source_tree_sha256": task.summary.sha256,
                "task_txt_sha256": task.task_txt_sha256,
            }
            for task in tasks
        ],
        "total_episodes": len(archive_records),
        "total_frames": sum(task.total_frames for task in tasks),
        "total_source_bytes": sum(task.summary.total_bytes for task in tasks),
        "total_tasks": len(tasks),
    }


def _render_readme(
    *,
    config: RawPackageConfig,
    tasks: tuple[RawTask, ...],
    contract: RawContract,
    archive_records: list[dict[str, Any]],
) -> str:
    total_frames = sum(task.total_frames for task in tasks)
    total_source_bytes = sum(task.summary.total_bytes for task in tasks)
    total_archive_bytes = sum(item["archive_bytes"] for item in archive_records)
    task_coverage = "<br>".join(
        f"{task.prompt} — {len(task.episodes)} episodes / {task.total_frames:,} frames"
        for task in tasks
    )
    camera_summary = "; ".join(
        f"`{camera['name']}` {camera['width']}×{camera['height']} "
        f"{str(camera['modality']).upper()}"
        for camera in contract.cameras
    )
    local_dir = config.repo_id.rsplit("/", maxsplit=1)[-1]
    return f"""---
pretty_name: NYUSH Galaxea A1 Fruit Placement Raw v3
tags:
  - robotics
  - manipulation
  - teleoperation
  - galaxea-a1
  - raw-dataset
---

# NYUSH Galaxea A1 — Fruit Placement Raw v3

![Annotated agent-view frame](assets/{config.readme_scene_image.name})

| Field | Value |
| --- | --- |
| Tasks | {task_coverage} |
| Total | {len(tasks)} tasks · {len(archive_records)} episodes · {total_frames:,} frames · {contract.fps_target:g} FPS |
| Cameras | {camera_summary} |
| State | {len(contract.state_names)} channels: end-effector pose, arm joints, continuous gripper |
| Action | {len(contract.action_names)} channels: absolute arm joints, continuous gripper |
| Format | `{TELEOP_RAW_SCHEMA_VERSION}` · one `.tar.zst` per episode |
| Payload | {_human_size(total_source_bytes)} raw · {_human_size(total_archive_bytes)} archived |

## Use

Download:

```bash
hf download {config.repo_id} --repo-type dataset --local-dir {local_dir}
cd {local_dir}
```

Verify the episode archives:

```bash
sha256sum -c checksums.sha256
```

Extract one episode:

```bash
mkdir -p restored
tar --use-compress-program=unzstd \
  -xf tasks/<task>/episodes/<episode>.tar.zst -C restored
```

Use `manifest.json` for the dataset summary and `episodes.jsonl` for per-episode
metadata and checksums. Convert the restored raw episodes to LeRobot v2.1 before
training.
"""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
            for value in values
        )
    )


def _write_checksums(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            f"{record['archive_sha256']}  {record['archive']}\n" for record in records
        )
    )


def _clean_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _required_metadata_string(
    data: dict[str, Any], key: str, metadata_path: Path
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metadata {key} must be a non-empty string: {metadata_path}")
    return value


def _metadata_names(
    data: dict[str, Any], key: str, metadata_path: Path
) -> tuple[str, ...]:
    value = data.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(
            f"metadata {key} must be a non-empty string list: {metadata_path}"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"metadata {key} contains duplicates: {metadata_path}")
    return tuple(value)


def _nonnegative_integer(data: dict[str, Any], key: str, metadata_path: Path) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"metadata {key} must be a nonnegative integer: {metadata_path}"
        )
    return value


def _positive_integer(data: dict[str, Any], key: str, metadata_path: Path) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"metadata {key} must be a positive integer: {metadata_path}")
    return value


def _human_size(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_raw_package_config(args.config)
        info(f"Config: {config.path}")
        info(f"Immutable sources: {len(config.source_roots)} task directories")
        info(f"Local review output: {config.output_root}")
        result = build_raw_package(config, progress=step)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        failure(str(exc))
        return 1
    success(
        f"Built {result.total_tasks} tasks, {result.total_episodes} episodes, "
        f"{result.total_frames} frames"
    )
    success(f"README ready for review: {result.readme_path}")
    info("No Hugging Face repository was created and no data was uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
