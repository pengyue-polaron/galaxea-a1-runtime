"""Derive a LingBot-VA EEF-action package from an A1 LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxea_a1_runtime.kinematics import SerialChainFK, compose_relative_pose, relative_pose
from galaxea_a1_runtime.lerobot.joint_pack import pack_joint_v3_dataset
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset

ACTION_NAMES = (
    "eef_delta_x_from_episode_start",
    "eef_delta_y_from_episode_start",
    "eef_delta_z_from_episode_start",
    "eef_delta_qx_from_episode_start",
    "eef_delta_qy_from_episode_start",
    "eef_delta_qz_from_episode_start",
    "eef_delta_qw_from_episode_start",
    "gripper_binary",
)
USED_ACTION_CHANNEL_IDS = (0, 1, 2, 3, 4, 5, 6, 28)
SOURCE_ACTION_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
SOURCE_STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)


@dataclass(frozen=True)
class LingBotPackConfig:
    source_root: Path
    v3_target_root: Path
    v3_archive_path: Path
    v3_repo_id: str
    v21_target_root: Path
    v21_archive_path: Path
    v21_repo_id: str
    joint_v3_target_root: Path
    joint_v3_archive_path: Path
    joint_v3_repo_id: str
    urdf_path: Path
    base_link: str
    tip_link: str
    gripper_stroke_scale_mm: float
    gripper_source_open_threshold: float
    gripper_policy_open_threshold: float


def load_pack_config(path: Path) -> LingBotPackConfig:
    config_path = path.expanduser().resolve()
    repo_root = config_path.parents[2]
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    dataset = raw["dataset"]
    outputs = raw["outputs"]
    v3 = outputs["v3"]
    v21 = outputs["v21"]
    joint_v3 = outputs["joint_v3"]
    kinematics = raw["kinematics"]
    gripper = raw["gripper"]
    return LingBotPackConfig(
        source_root=_repo_path(repo_root, dataset["source_root"]),
        v3_target_root=_repo_path(repo_root, v3["target_root"]),
        v3_archive_path=_repo_path(repo_root, v3["archive_path"]),
        v3_repo_id=str(v3["repo_id"]),
        v21_target_root=_repo_path(repo_root, v21["target_root"]),
        v21_archive_path=_repo_path(repo_root, v21["archive_path"]),
        v21_repo_id=str(v21["repo_id"]),
        joint_v3_target_root=_repo_path(repo_root, joint_v3["target_root"]),
        joint_v3_archive_path=_repo_path(repo_root, joint_v3["archive_path"]),
        joint_v3_repo_id=str(joint_v3["repo_id"]),
        urdf_path=_repo_path(repo_root, kinematics["urdf"]),
        base_link=str(kinematics["base_link"]),
        tip_link=str(kinematics["tip_link"]),
        gripper_stroke_scale_mm=float(gripper["stroke_scale_mm"]),
        gripper_source_open_threshold=float(gripper["source_open_threshold"]),
        gripper_policy_open_threshold=float(gripper["policy_open_threshold"]),
    )


def pack_lingbot_dataset(
    *,
    source_root: Path,
    target_root: Path,
    urdf_path: Path,
    repo_id: str,
    gripper_stroke_scale_mm: float = 200.0,
    base_link: str = "base_link",
    tip_link: str = "arm_seg6",
    gripper_source_open_threshold: float = 0.15,
    gripper_policy_open_threshold: float = 0.5,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    info = _load_json(source_root / "meta/info.json")
    _validate_source(info)
    if gripper_stroke_scale_mm <= 0:
        raise ValueError("gripper_stroke_scale_mm must be positive")
    for label, threshold in (
        ("gripper_source_open_threshold", gripper_source_open_threshold),
        ("gripper_policy_open_threshold", gripper_policy_open_threshold),
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"{label} must be between 0 and 1")
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"target root exists: {target_root}")
        shutil.rmtree(target_root)

    chain = SerialChainFK.from_urdf(urdf_path, base_link=base_link, tip_link=tip_link)
    expected_joints = tuple(f"arm_joint{index}" for index in range(1, 7))
    if chain.joint_names != expected_joints:
        raise ValueError(f"unexpected A1 URDF chain: {chain.joint_names}")

    target_root.mkdir(parents=True)
    _copy_tree_with_hardlinks(source_root, target_root)
    data_files = sorted(target_root.glob("data/**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no LeRobot parquet data under {source_root}")

    episode_actions: dict[int, np.ndarray] = {}
    episode_state_values: dict[int, np.ndarray] = {}
    source_hash = hashlib.sha256()
    fk_feedback_position_errors: list[np.ndarray] = []
    fk_feedback_quat_dots: list[np.ndarray] = []
    reconstructed_position_errors: list[np.ndarray] = []
    reconstructed_quat_dots: list[np.ndarray] = []

    for path in data_files:
        frame = pd.read_parquet(path)
        source_hash.update((source_root / path.relative_to(target_root)).read_bytes())
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        joint_actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        converted = np.empty((len(frame), 8), dtype=np.float32)
        converted_states = states.copy().astype(np.float32)

        for episode_index in frame["episode_index"].drop_duplicates().tolist():
            mask = frame["episode_index"].to_numpy() == episode_index
            episode_observations = states[mask]
            episode_joint_actions = joint_actions[mask]
            initial_pose = episode_observations[0, :7]
            feedback_fk = np.stack([chain.pose(values) for values in episode_observations[:, 7:13]])
            target_fk = np.stack([chain.pose(values) for values in episode_joint_actions[:, :6]])
            _align_quaternion_signs(feedback_fk[:, 3:7], episode_observations[:, 3:7])
            fk_feedback_position_errors.append(
                np.linalg.norm(feedback_fk[:, :3] - episode_observations[:, :3], axis=1)
            )
            fk_feedback_quat_dots.append(
                np.abs(np.sum(feedback_fk[:, 3:7] * episode_observations[:, 3:7], axis=1))
            )

            episode_converted = np.empty((len(episode_observations), 8), dtype=np.float64)
            for row, target_pose in enumerate(target_fk):
                episode_converted[row, :7] = relative_pose(target_pose, initial_pose)
            _make_quaternions_continuous(episode_converted[:, 3:7])
            source_gripper = episode_joint_actions[:, 6]
            if np.any(source_gripper < -1e-6) or np.any(source_gripper > 1.0 + 1e-6):
                raise ValueError(f"episode {episode_index} gripper action is outside normalized [0, 1]")
            episode_converted[:, 7] = (source_gripper >= gripper_source_open_threshold).astype(np.float64)
            if int(episode_index) in episode_state_values:
                raise ValueError(f"episode {episode_index} appears in more than one data file")
            binary_state = episode_observations.astype(np.float32, copy=True)
            binary_state[:, -1] = (
                episode_observations[:, -1] >= gripper_source_open_threshold
            ).astype(np.float32)
            converted_states[mask] = binary_state

            reconstructed = np.stack(
                [compose_relative_pose(values[:7], initial_pose) for values in episode_converted]
            )
            _align_quaternion_signs(reconstructed[:, 3:7], target_fk[:, 3:7])
            reconstructed_position_errors.append(np.linalg.norm(reconstructed[:, :3] - target_fk[:, :3], axis=1))
            reconstructed_quat_dots.append(np.abs(np.sum(reconstructed[:, 3:7] * target_fk[:, 3:7], axis=1)))
            converted[mask] = episode_converted.astype(np.float32)
            episode_actions[int(episode_index)] = episode_converted.astype(np.float32)
            episode_state_values[int(episode_index)] = binary_state

        frame["observation.state"] = list(converted_states)
        frame["action"] = list(converted)
        frame.to_parquet(path, index=False)

    all_actions = np.concatenate([episode_actions[index] for index in sorted(episode_actions)])
    all_states = np.concatenate(
        [episode_state_values[index] for index in sorted(episode_state_values)]
    )
    global_action_stats = _vector_stats(all_actions)
    global_state_stats = _vector_stats(all_states)
    _rewrite_info(target_root, info)
    _rewrite_global_stats(target_root, global_action_stats, global_state_stats)
    _rewrite_episode_stats(target_root, episode_actions, episode_state_values)

    validation = {
        "fk_feedback_position_error_m": _error_summary(np.concatenate(fk_feedback_position_errors)),
        "fk_feedback_min_abs_quaternion_dot": float(np.min(np.concatenate(fk_feedback_quat_dots))),
        "roundtrip_position_error_m": _error_summary(np.concatenate(reconstructed_position_errors)),
        "roundtrip_min_abs_quaternion_dot": float(np.min(np.concatenate(reconstructed_quat_dots))),
        "binary_gripper": {
            "action_unique_values": np.unique(all_actions[:, -1]).tolist(),
            "state_unique_values": np.unique(all_states[:, -1]).tolist(),
            "closed_action_frames": int(np.count_nonzero(all_actions[:, -1] == 0.0)),
            "open_action_frames": int(np.count_nonzero(all_actions[:, -1] == 1.0)),
            "action_transitions": _count_episode_transitions(episode_actions),
        },
    }
    manifest = {
        "format": "lerobot_v3_lingbot_va_a1_eef_binary_v1",
        "repo_id": repo_id,
        "source_dataset": str(source_root),
        "source_data_sha256": source_hash.hexdigest(),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "robot": "galaxea_a1",
        "kinematics": {
            "urdf": str(urdf_path),
            "urdf_sha256": _file_sha256(urdf_path),
            "base_link": base_link,
            "tip_link": tip_link,
            "joint_names": list(chain.joint_names),
        },
        "action": {
            "shape": [8],
            "names": list(ACTION_NAMES),
            "semantics": "RoboTwin-style EEF target relative to episode initial feedback pose",
            "translation": "target_xyz_base_link - initial_xyz_base_link",
            "rotation": "inverse(initial_quaternion) * target_quaternion, xyzw",
            "target_source": "FK of same-frame joint_absolute action",
            "gripper": "binary: closed=0, open=1; no intermediate values",
            "gripper_physical_mapping": f"0 -> 0 mm, 1 -> {gripper_stroke_scale_mm:g} mm",
            "gripper_stroke_scale_mm": gripper_stroke_scale_mm,
            "source_open_threshold": gripper_source_open_threshold,
            "lingbot_used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        },
        "cameras": {
            "ordered_keys": ["observation.images.front", "observation.images.wrist"],
            "layout": "width_concat",
        },
        "recommended_policy": {
            "checkpoint": "lerobot/lingbot_va_robotwin",
            "use_peft": True,
            "attn_mode": "flex",
            "obs_cam_keys": ["observation.images.front", "observation.images.wrist"],
            "camera_layout": "width_concat",
            "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
            "runtime_gripper_stroke_scale_mm": gripper_stroke_scale_mm,
            "runtime_gripper_open_threshold": gripper_policy_open_threshold,
            "action_per_frame": 4,
            "frame_chunk_size": 4,
        },
        "validation": validation,
    }
    _write_json(target_root / "meta/lingbot_va.json", manifest)
    (target_root / "TRAINING.md").write_text(
        _training_doc(gripper_stroke_scale_mm), encoding="utf-8"
    )
    package_sha256 = _dataset_digest(target_root)
    manifest["package_sha256"] = package_sha256
    _write_json(target_root / "meta/lingbot_va.json", manifest)

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
        raise ValueError("source must be a LeRobot v3.0 dataset")
    features = info.get("features", {})
    if features.get("action", {}).get("names") != list(SOURCE_ACTION_NAMES):
        raise ValueError("source action must be A1 joint_absolute + normalized gripper")
    if features.get("observation.state", {}).get("names") != list(SOURCE_STATE_NAMES):
        raise ValueError("source observation.state does not contain the expected A1 EEF and joints")
    for key in ("observation.images.front", "observation.images.wrist"):
        if key not in features:
            raise ValueError(f"source is missing required camera feature {key!r}")


def _copy_tree_with_hardlinks(source: Path, target: Path) -> None:
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        if relative.parts[0] == "images":
            continue
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
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
    info["robot_type"] = "galaxea_a1_lingbot_eef"
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [8],
        "names": list(ACTION_NAMES),
    }
    info["features"]["observation.state"]["names"][-1] = "gripper_binary"
    _write_json(target_root / "meta/info.json", info)


def _rewrite_global_stats(
    target_root: Path,
    action_stats: dict[str, list[float]],
    state_stats: dict[str, list[float]],
) -> None:
    path = target_root / "meta/stats.json"
    stats = _load_json(path)
    stats["action"] = action_stats
    stats["observation.state"] = state_stats
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
                    column = f"stats/{feature}/{statistic}"
                    episodes.at[row_index, column] = statistic_values
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


def _count_episode_transitions(episodes: dict[int, np.ndarray]) -> int:
    return sum(
        int(np.count_nonzero(np.diff(episodes[index][:, -1])))
        for index in sorted(episodes)
    )


def _align_quaternion_signs(values: np.ndarray, references: np.ndarray) -> None:
    signs = np.sum(values * references, axis=1) < 0
    values[signs] *= -1


def _make_quaternions_continuous(values: np.ndarray) -> None:
    if values[0, 3] < 0:
        values[0] *= -1
    for index in range(1, len(values)):
        if float(np.dot(values[index - 1], values[index])) < 0:
            values[index] *= -1


def _error_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _dataset_digest(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {Path("meta/lingbot_va.json")}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative in excluded:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_doc(gripper_stroke_scale_mm: float) -> str:
    return f"""# A1 LingBot-VA Dataset

The two ordered observations are the external `front` camera followed by the eye-in-hand
`wrist` camera. See `meta/lingbot_va.json` for the exact
EEF action convention, source hash, URDF hash, channel mapping, and validation results.

Use the RoboTwin checkpoint as the action-semantics baseline, LoRA/PEFT training, camera layout
`width_concat`, and action channels `[0,1,2,3,4,5,6,28]`. At inference, compose each predicted
EEF delta onto the episode's measured initial `base_link -> arm_seg6` pose before publishing the
absolute target. Gripper is strictly binary: closed=0 and open=1. The hardware adapter maps these
states to 0 mm and {gripper_stroke_scale_mm:g} mm respectively.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/banana_in_the_plate.toml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_pack_config(args.config)
    v3_manifest = pack_lingbot_dataset(
        source_root=config.source_root,
        target_root=config.v3_target_root,
        urdf_path=config.urdf_path,
        repo_id=config.v3_repo_id,
        gripper_stroke_scale_mm=config.gripper_stroke_scale_mm,
        base_link=config.base_link,
        tip_link=config.tip_link,
        gripper_source_open_threshold=config.gripper_source_open_threshold,
        gripper_policy_open_threshold=config.gripper_policy_open_threshold,
        overwrite=args.overwrite,
        archive_path=config.v3_archive_path,
    )
    v21_manifest = export_v21_dataset(
        source_root=config.v3_target_root,
        target_root=config.v21_target_root,
        repo_id=config.v21_repo_id,
        overwrite=args.overwrite,
        archive_path=config.v21_archive_path,
    )
    joint_v3_manifest = pack_joint_v3_dataset(
        source_root=config.source_root,
        target_root=config.joint_v3_target_root,
        repo_id=config.joint_v3_repo_id,
        gripper_open_threshold=config.gripper_source_open_threshold,
        overwrite=args.overwrite,
        archive_path=config.joint_v3_archive_path,
    )
    print(
        json.dumps(
            {
                "eef_v3.0": v3_manifest,
                "eef_v2.1": v21_manifest,
                "joint_v3.0": joint_v3_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
