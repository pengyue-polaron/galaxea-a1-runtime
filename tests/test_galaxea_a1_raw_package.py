import csv
import json
import shutil
from pathlib import Path

import pytest

from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.datasets.raw_package import (
    build_raw_package,
    load_raw_package_config,
)


def _write_episode(
    task_root: Path,
    *,
    task: str,
    episode_index: int = 0,
    collection_suffix: str = "",
) -> None:
    episode = task_root / f"episode_{episode_index:03d}{collection_suffix}"
    (episode / "cam0").mkdir(parents=True)
    (episode / "cam1").mkdir()
    (episode / "cam0" / "000000.jpg").write_bytes(b"front-image")
    (episode / "cam1" / "000000.jpg").write_bytes(b"wrist-image")
    metadata = {
        "schema_version": TELEOP_RAW_SCHEMA_VERSION,
        "collection_mode": "teleop",
        "task": task,
        "experiment": task_root.name,
        "episode_index": episode_index,
        "frame_count": 1,
        "fps_target": 30.0,
        "state_mode": "eef_joint",
        "action_mode": "joint_absolute",
        "state_names": ["eef_x", "gripper"],
        "action_names": ["joint_1", "gripper"],
        "state_topics": {},
        "action_topics": {},
        "control_path": [],
        "config_path": "configs/teleop/test.toml",
        "quality_checks": {},
        "cameras": [
            {
                "name": "front",
                "directory": "cam0",
                "width": 2,
                "height": 2,
                "modality": "rgb",
                "encoding": "bgr8_jpeg",
            },
            {
                "name": "wrist",
                "directory": "cam1",
                "width": 2,
                "height": 2,
                "modality": "rgb",
                "encoding": "bgr8_jpeg",
            },
        ],
    }
    (episode / "metadata.json").write_text(json.dumps(metadata))
    with (episode / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "cam0_relpath", "cam1_relpath"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_index": 0,
                "cam0_relpath": "cam0/000000.jpg",
                "cam1_relpath": "cam1/000000.jpg",
            }
        )


def _write_task(repo_root: Path, slug: str, prompt: str) -> Path:
    task_root = repo_root / "data" / "raw" / slug
    task_root.mkdir(parents=True)
    (task_root / "task.txt").write_text(prompt + "\n")
    _write_episode(task_root, task=prompt)
    return task_root


def _write_config(repo_root: Path, source_slugs: list[str], **changes) -> Path:
    config_dir = repo_root / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    scene_image = repo_root / "assets/datasets/scene.png"
    scene_image.parent.mkdir(parents=True, exist_ok=True)
    scene_image.write_bytes(b"annotated-scene")
    values = {
        "repo_id": "owner/fruit-placement-raw-v3",
        "output_root": "data/exports/fruit-placement-raw-v3",
        "readme_scene_image": "assets/datasets/scene.png",
        "source_roots": [f"data/raw/{slug}" for slug in source_slugs],
        "archive_format": "tar.zst",
        "zstd_level": 3,
        "overwrite": False,
    }
    values.update(changes)
    lines = ["[raw_package]"]
    for key, value in values.items():
        if isinstance(value, str):
            rendered = json.dumps(value)
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = json.dumps(value)
        lines.append(f"{key} = {rendered}")
    path = config_dir / "raw.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_raw_package_config_is_strict_and_scoped(tmp_path):
    _write_task(tmp_path, "put-fruit-in-bowl", "Put fruit in bowl")
    path = _write_config(tmp_path, ["put-fruit-in-bowl"])

    config = load_raw_package_config(path, repo_root=tmp_path)

    assert config.output_root == tmp_path / "data/exports/fruit-placement-raw-v3"
    assert config.readme_scene_image == tmp_path / "assets/datasets/scene.png"
    assert config.source_roots == (tmp_path / "data/raw/put-fruit-in-bowl",)

    text = path.read_text().replace(
        "overwrite = false", "overwrite = false\nunknown = true"
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="unknown"):
        load_raw_package_config(path, repo_root=tmp_path)


def test_raw_package_rejects_source_outside_raw_root(tmp_path):
    path = _write_config(
        tmp_path,
        ["unused"],
        source_roots=["outside/source"],
    )

    with pytest.raises(ValueError, match="must be below"):
        load_raw_package_config(path, repo_root=tmp_path)


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("tar", "zstd", "unzstd")),
    reason="archive tools are not installed",
)
def test_raw_package_uses_copy_and_verifies_round_trip(tmp_path):
    source = _write_task(tmp_path, "put-fruit-in-bowl", "Put fruit in bowl")
    config_path = _write_config(tmp_path, [source.name])
    config = load_raw_package_config(config_path, repo_root=tmp_path)
    source_before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    result = build_raw_package(config)

    source_after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before
    assert result.total_tasks == 1
    assert result.total_episodes == 1
    assert result.total_frames == 1
    readme = result.readme_path.read_text()
    assert "Private internal dataset" in readme
    assert "license:" not in readme
    assert "![Annotated agent-view frame](assets/scene.png)" in readme
    assert (result.output_root / "assets/scene.png").read_bytes() == b"annotated-scene"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["raw_schema_version"] == TELEOP_RAW_SCHEMA_VERSION
    assert manifest["repo_id"] == "owner/fruit-placement-raw-v3"
    assert manifest["readme_scene_image"]["path"] == "assets/scene.png"
    assert not any(
        "staging" in path.name for path in config.output_root.parent.iterdir()
    )

    archive = next(config.output_root.glob("tasks/*/episodes/*.tar.zst"))
    restored = tmp_path / "restored"
    restored.mkdir()
    result = shutil.which("tar")
    assert result is not None
    import subprocess

    subprocess.run(
        [
            result,
            "--use-compress-program=unzstd",
            "-xf",
            str(archive),
            "-C",
            str(restored),
        ],
        check=True,
    )
    restored_episode = restored / "episode_000"
    assert (restored_episode / "cam0/000000.jpg").read_bytes() == b"front-image"


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("tar", "zstd", "unzstd")),
    reason="archive tools are not installed",
)
def test_raw_package_preserves_collection_suffix_in_episode_name(tmp_path):
    task_root = tmp_path / "data/raw/put-fruit-in-bowl"
    task_root.mkdir(parents=True)
    (task_root / "task.txt").write_text("Put fruit in bowl\n")
    _write_episode(
        task_root,
        task="Put fruit in bowl",
        collection_suffix="_20260715_184450",
    )
    config = load_raw_package_config(
        _write_config(tmp_path, [task_root.name]), repo_root=tmp_path
    )

    result = build_raw_package(config)

    assert next(result.output_root.glob("tasks/*/episodes/*.tar.zst")).name == (
        "episode_000_20260715_184450.tar.zst"
    )


def test_raw_package_rejects_symlinks(tmp_path):
    source = _write_task(tmp_path, "put-fruit-in-bowl", "Put fruit in bowl")
    (source / "episode_000" / "link").symlink_to(source / "task.txt")
    config = load_raw_package_config(
        _write_config(tmp_path, [source.name]), repo_root=tmp_path
    )

    with pytest.raises(ValueError, match="symlinks"):
        build_raw_package(config)


def test_raw_package_stale_staging_blocks_retry(tmp_path):
    source = _write_task(tmp_path, "put-fruit-in-bowl", "Put fruit in bowl")
    config = load_raw_package_config(
        _write_config(tmp_path, [source.name]), repo_root=tmp_path
    )
    config.output_root.parent.mkdir(parents=True)
    stale = config.output_root.parent / f".{config.output_root.name}.staging-crash"
    stale.mkdir()

    with pytest.raises(RuntimeError, match="operator inspection"):
        build_raw_package(config)
