"""Build a joint-action LeRobot v3 package from the selected A1 dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JOINT_ACTION_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_normalized",
)


def pack_joint_v3_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    info = _read_json(source_root / "meta/info.json")
    _validate_source(info)
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"target root exists: {target_root}")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)
    _copy_tree_with_video_hardlinks(source_root, target_root)

    episode_actions: dict[int, np.ndarray] = {}
    episode_states: dict[int, np.ndarray] = {}
    for path in sorted(target_root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(path)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        _validate_normalized_gripper(actions[:, -1], label="action")
        _validate_normalized_gripper(states[:, -1], label="observation.state")
        for episode_index in frame["episode_index"].drop_duplicates().tolist():
            key = int(episode_index)
            if key in episode_actions:
                raise ValueError(f"episode {key} appears in more than one data file")
            mask = frame["episode_index"].to_numpy() == episode_index
            episode_actions[key] = actions[mask]
            episode_states[key] = states[mask]
        frame["action"] = list(actions)
        frame["observation.state"] = list(states)
        frame.to_parquet(path, index=False)

    all_actions = np.concatenate([episode_actions[index] for index in sorted(episode_actions)])
    all_states = np.concatenate([episode_states[index] for index in sorted(episode_states)])
    _rewrite_info(target_root, info)
    _rewrite_global_stats(target_root, all_actions, all_states)
    _rewrite_episode_stats(target_root, episode_actions, episode_states)

    manifest = {
        "format": "lerobot_v3_galaxea_a1_joint_continuous_v1",
        "repo_id": repo_id,
        "source_dataset": str(source_root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "observation": {
            "shape": [14],
            "semantics": "EEF pose (xyz+quaternion), six measured joints, continuous normalized gripper",
            "joint_unit": "radian",
        },
        "action": {
            "shape": [7],
            "names": list(JOINT_ACTION_NAMES),
            "semantics": "six absolute A1 joint targets plus continuous normalized gripper target",
            "joint_unit": "radian",
            "gripper": "continuous normalized target: 0=minimum stroke, 1=maximum stroke",
        },
        "cameras": {
            "ordered_keys": ["observation.images.front", "observation.images.wrist"],
        },
        "validation": {
            "action_gripper_min": float(np.min(all_actions[:, -1])),
            "action_gripper_max": float(np.max(all_actions[:, -1])),
            "state_gripper_min": float(np.min(all_states[:, -1])),
            "state_gripper_max": float(np.max(all_states[:, -1])),
            "intermediate_action_frames": int(
                np.count_nonzero((all_actions[:, -1] > 0.0) & (all_actions[:, -1] < 1.0))
            ),
        },
    }
    _write_json(target_root / "meta/joint_v3.json", manifest)
    (target_root / "TRAINING.md").write_text(
        "# A1 Joint LeRobot Dataset\n\n"
        "Action is `[joint_1..joint_6, gripper]`. Joint values are absolute targets in radians. "
        "Gripper is continuous and normalized: `0=minimum stroke`, `1=maximum stroke`.\n",
        encoding="utf-8",
    )
    manifest["package_sha256"] = _dataset_digest(target_root, exclude={Path("meta/joint_v3.json")})
    _write_json(target_root / "meta/joint_v3.json", manifest)

    if archive_path is not None:
        archive_path = archive_path.expanduser().resolve()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(target_root, arcname=target_root.name)
        manifest["archive"] = str(archive_path)
        manifest["archive_sha256"] = _file_sha256(archive_path)
        archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
            f"{manifest['archive_sha256']}  {archive_path.name}\n", encoding="ascii"
        )
    return manifest


def _validate_source(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError("joint package source must be a LeRobot v3.0 dataset")
    action = info.get("features", {}).get("action", {})
    if action.get("names") != [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]:
        raise ValueError("joint package source must contain six A1 joint actions and gripper")


def _copy_tree_with_video_hardlinks(source: Path, target: Path) -> None:
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        if relative.parts[0] == "images":
            continue
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts[0] != "videos":
            shutil.copy2(source_path, target_path)
            continue
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def _rewrite_info(target_root: Path, source_info: dict[str, Any]) -> None:
    info = json.loads(json.dumps(source_info))
    info["robot_type"] = "galaxea_a1_joint"
    info["features"]["action"]["names"] = list(JOINT_ACTION_NAMES)
    state_names = info["features"]["observation.state"]["names"]
    state_names[7:13] = [f"joint_{index}_rad" for index in range(1, 7)]
    state_names[-1] = "gripper_normalized"
    _write_json(target_root / "meta/info.json", info)


def _rewrite_global_stats(target_root: Path, actions: np.ndarray, states: np.ndarray) -> None:
    path = target_root / "meta/stats.json"
    stats = _read_json(path)
    stats["action"] = _vector_stats(actions)
    stats["observation.state"] = _vector_stats(states)
    _write_json(path, stats)


def _rewrite_episode_stats(
    target_root: Path,
    episode_actions: dict[int, np.ndarray],
    episode_states: dict[int, np.ndarray],
) -> None:
    for path in sorted(target_root.glob("meta/episodes/**/*.parquet")):
        episodes = pd.read_parquet(path)
        for row_index, episode_index in enumerate(episodes["episode_index"].to_numpy()):
            for feature, values in (
                ("action", episode_actions[int(episode_index)]),
                ("observation.state", episode_states[int(episode_index)]),
            ):
                for statistic, statistic_values in _vector_stats(values).items():
                    episodes.at[row_index, f"stats/{feature}/{statistic}"] = statistic_values
        episodes.to_parquet(path, index=False)


def _vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "count": [int(len(x))],
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q10": np.quantile(x, 0.10, axis=0).tolist(),
        "q50": np.quantile(x, 0.50, axis=0).tolist(),
        "q90": np.quantile(x, 0.90, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _validate_normalized_gripper(values: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} gripper contains non-finite values")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(f"{label} gripper is outside normalized [0, 1]")


def _dataset_digest(root: Path, *, exclude: set[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative in exclude:
            continue
        digest.update(str(relative).encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
