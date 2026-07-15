"""Build a joint-action LeRobot v3 package from the selected A1 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxea_a1_runtime.filesystem import (
    atomic_output_directory,
)
from galaxea_a1_runtime.lerobot.dataset_package import (
    copy_dataset_tree,
    dataset_digest,
    read_json,
    rewrite_episode_vector_stats,
    vector_stats,
    write_json,
    write_tar_archive,
)
from galaxea_a1_runtime.schema import (
    DEFAULT_RGB_IMAGE_KEYS,
    DEFAULT_STATE_NAMES,
    JOINT_ACTION_NAMES as SOURCE_ACTION_NAMES,
    JOINT_ACTION_NAMES_RAD as TARGET_ACTION_NAMES,
)


def pack_joint_v3_dataset(
    *,
    source_root: Path,
    target_root: Path,
    repo_id: str,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    final_target_root = target_root.expanduser().resolve()
    with atomic_output_directory(
        final_target_root, overwrite=overwrite
    ) as staging_root:
        return _build_joint_v3_dataset(
            source_root=source_root,
            target_root=staging_root,
            final_target_root=final_target_root,
            repo_id=repo_id,
            archive_path=archive_path,
        )


def _build_joint_v3_dataset(
    *,
    source_root: Path,
    target_root: Path,
    final_target_root: Path,
    repo_id: str,
    archive_path: Path | None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    info = read_json(source_root / "meta/info.json")
    _validate_source(info)
    copy_dataset_tree(source_root, target_root)

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

    all_actions = np.concatenate(
        [episode_actions[index] for index in sorted(episode_actions)]
    )
    all_states = np.concatenate(
        [episode_states[index] for index in sorted(episode_states)]
    )
    _rewrite_info(target_root, info)
    _rewrite_global_stats(target_root, all_actions, all_states)
    rewrite_episode_vector_stats(
        target_root,
        episode_actions=episode_actions,
        episode_states=episode_states,
    )

    manifest = {
        "format": "lerobot_v3_galaxea_a1_joint_continuous_v1",
        "repo_id": repo_id,
        "source_dataset": str(source_root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "observation": {
            "shape": [len(DEFAULT_STATE_NAMES)],
            "semantics": "EEF pose (xyz+quaternion), six measured joints, continuous normalized gripper",
            "joint_unit": "radian",
        },
        "action": {
            "shape": [len(TARGET_ACTION_NAMES)],
            "names": list(TARGET_ACTION_NAMES),
            "semantics": "six absolute A1 joint targets plus continuous normalized gripper target",
            "joint_unit": "radian",
            "gripper": "continuous normalized target: 0=minimum stroke, 1=maximum stroke",
        },
        "cameras": {
            "ordered_keys": list(DEFAULT_RGB_IMAGE_KEYS),
        },
        "validation": {
            "action_gripper_min": float(np.min(all_actions[:, -1])),
            "action_gripper_max": float(np.max(all_actions[:, -1])),
            "state_gripper_min": float(np.min(all_states[:, -1])),
            "state_gripper_max": float(np.max(all_states[:, -1])),
            "intermediate_action_frames": int(
                np.count_nonzero(
                    (all_actions[:, -1] > 0.0) & (all_actions[:, -1] < 1.0)
                )
            ),
        },
    }
    write_json(target_root / "meta/joint_v3.json", manifest)
    (target_root / "TRAINING.md").write_text(
        "# A1 Joint LeRobot Dataset\n\n"
        "Action is `[joint_1..joint_6, gripper]`. Joint values are absolute targets in radians. "
        "Gripper is continuous and normalized: `0=minimum stroke`, `1=maximum stroke`.\n",
        encoding="utf-8",
    )
    manifest["package_sha256"] = dataset_digest(
        target_root, exclude={Path("meta/joint_v3.json")}
    )
    write_json(target_root / "meta/joint_v3.json", manifest)

    if archive_path is not None:
        archive_path, archive_sha256 = write_tar_archive(
            target_root,
            archive_path=archive_path,
            root_name=final_target_root.name,
        )
        manifest["archive"] = str(archive_path)
        manifest["archive_sha256"] = archive_sha256
    return manifest


def _validate_source(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError("joint package source must be a LeRobot v3.0 dataset")
    action = info.get("features", {}).get("action", {})
    if action.get("names") != list(SOURCE_ACTION_NAMES):
        raise ValueError(
            "joint package source must contain six A1 joint actions and gripper"
        )


def _rewrite_info(target_root: Path, source_info: dict[str, Any]) -> None:
    info = json.loads(json.dumps(source_info))
    info["robot_type"] = "galaxea_a1_joint"
    info["features"]["action"]["names"] = list(TARGET_ACTION_NAMES)
    state_names = info["features"]["observation.state"]["names"]
    eef_state_dof = len(DEFAULT_STATE_NAMES) - len(SOURCE_ACTION_NAMES)
    arm_dof = len(SOURCE_ACTION_NAMES) - 1
    state_names[eef_state_dof : eef_state_dof + arm_dof] = list(
        TARGET_ACTION_NAMES[:-1]
    )
    state_names[-1] = "gripper_normalized"
    write_json(target_root / "meta/info.json", info)


def _rewrite_global_stats(
    target_root: Path, actions: np.ndarray, states: np.ndarray
) -> None:
    path = target_root / "meta/stats.json"
    stats = read_json(path)
    stats["action"] = vector_stats(actions)
    stats["observation.state"] = vector_stats(states)
    write_json(path, stats)


def _validate_normalized_gripper(values: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} gripper contains non-finite values")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(f"{label} gripper is outside normalized [0, 1]")
