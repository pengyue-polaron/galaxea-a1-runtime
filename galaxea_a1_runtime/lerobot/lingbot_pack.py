"""Build A1 LeRobot v2.1/v3.0, LingBot-VA, and ACT dataset packages."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxea_a1_runtime.console import ArgumentParser

from galaxea_a1_runtime.kinematics import (
    SerialChainFK,
    compose_relative_pose,
    relative_pose,
)
from galaxea_a1_runtime.filesystem import (
    atomic_output_directory,
)
from galaxea_a1_runtime.lerobot.dataset_package import (
    copy_dataset_tree,
    dataset_digest,
    file_sha256,
    read_json,
    rewrite_episode_vector_stats,
    vector_stats,
    write_json,
    write_tar_archive,
)
from galaxea_a1_runtime.lerobot.convert_raw import convert_raw_dataset
from galaxea_a1_runtime.lerobot.joint_pack import pack_joint_v3_dataset
from galaxea_a1_runtime.lerobot.lingbot_pack_config import load_pack_config
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset
from galaxea_a1_runtime.schema import (
    DEFAULT_RGB_IMAGE_KEYS,
    DEFAULT_STATE_NAMES,
    JOINT_ACTION_NAMES,
    LINGBOT_EEF_ACTION_CHANNEL_IDS,
)

ACTION_NAMES = (
    "eef_delta_x_from_episode_start",
    "eef_delta_y_from_episode_start",
    "eef_delta_z_from_episode_start",
    "eef_delta_qx_from_episode_start",
    "eef_delta_qy_from_episode_start",
    "eef_delta_qz_from_episode_start",
    "eef_delta_qw_from_episode_start",
    "gripper_normalized",
)
SOURCE_ACTION_NAMES = JOINT_ACTION_NAMES
SOURCE_STATE_NAMES = DEFAULT_STATE_NAMES


@dataclass
class _ConversionResult:
    episode_actions: dict[int, np.ndarray]
    episode_states: dict[int, np.ndarray]
    source_data_sha256: str
    fk_position_errors: list[np.ndarray]
    fk_quaternion_dots: list[np.ndarray]
    roundtrip_position_errors: list[np.ndarray]
    roundtrip_quaternion_dots: list[np.ndarray]


def pack_lingbot_dataset(
    *,
    source_root: Path,
    target_root: Path,
    urdf_path: Path,
    repo_id: str,
    gripper_stroke_min_mm: float,
    gripper_stroke_max_mm: float,
    base_link: str,
    tip_link: str,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    final_target_root = target_root.expanduser().resolve()
    with atomic_output_directory(
        final_target_root, overwrite=overwrite
    ) as staging_root:
        return _build_lingbot_dataset(
            source_root=source_root,
            target_root=staging_root,
            final_target_root=final_target_root,
            urdf_path=urdf_path,
            repo_id=repo_id,
            gripper_stroke_min_mm=gripper_stroke_min_mm,
            gripper_stroke_max_mm=gripper_stroke_max_mm,
            base_link=base_link,
            tip_link=tip_link,
            archive_path=archive_path,
        )


def _build_lingbot_dataset(
    *,
    source_root: Path,
    target_root: Path,
    final_target_root: Path,
    urdf_path: Path,
    repo_id: str,
    gripper_stroke_min_mm: float,
    gripper_stroke_max_mm: float,
    base_link: str,
    tip_link: str,
    archive_path: Path | None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    info = read_json(source_root / "meta/info.json")
    _validate_source(info)
    if gripper_stroke_max_mm <= gripper_stroke_min_mm:
        raise ValueError("gripper stroke maximum must be greater than minimum")
    chain = SerialChainFK.from_urdf(urdf_path, base_link=base_link, tip_link=tip_link)
    expected_joints = tuple(f"arm_joint{index}" for index in range(1, 7))
    if chain.joint_names != expected_joints:
        raise ValueError(f"unexpected A1 URDF chain: {chain.joint_names}")

    copy_dataset_tree(source_root, target_root)
    data_files = sorted(target_root.glob("data/**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no LeRobot parquet data under {source_root}")

    converted = _convert_data_files(
        data_files=data_files,
        source_root=source_root,
        target_root=target_root,
        chain=chain,
    )
    episode_actions = converted.episode_actions
    episode_state_values = converted.episode_states

    all_actions = np.concatenate(
        [episode_actions[index] for index in sorted(episode_actions)]
    )
    all_states = np.concatenate(
        [episode_state_values[index] for index in sorted(episode_state_values)]
    )
    global_action_stats = vector_stats(all_actions)
    global_state_stats = vector_stats(all_states)
    _rewrite_info(target_root, info)
    _rewrite_global_stats(target_root, global_action_stats, global_state_stats)
    rewrite_episode_vector_stats(
        target_root,
        episode_actions=episode_actions,
        episode_states=episode_state_values,
    )

    validation = {
        "fk_feedback_position_error_m": _error_summary(
            np.concatenate(converted.fk_position_errors)
        ),
        "fk_feedback_min_abs_quaternion_dot": float(
            np.min(np.concatenate(converted.fk_quaternion_dots))
        ),
        "roundtrip_position_error_m": _error_summary(
            np.concatenate(converted.roundtrip_position_errors)
        ),
        "roundtrip_min_abs_quaternion_dot": float(
            np.min(np.concatenate(converted.roundtrip_quaternion_dots))
        ),
        "continuous_gripper": {
            "action_min": float(np.min(all_actions[:, -1])),
            "action_max": float(np.max(all_actions[:, -1])),
            "state_min": float(np.min(all_states[:, -1])),
            "state_max": float(np.max(all_states[:, -1])),
            "intermediate_action_frames": int(
                np.count_nonzero(
                    (all_actions[:, -1] > 0.0) & (all_actions[:, -1] < 1.0)
                )
            ),
        },
    }
    manifest = {
        "format": "lerobot_v3_lingbot_va_a1_eef_continuous_v1",
        "repo_id": repo_id,
        "source_dataset": str(source_root),
        "source_data_sha256": converted.source_data_sha256,
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "robot": "galaxea_a1",
        "kinematics": {
            "urdf": str(urdf_path),
            "urdf_sha256": file_sha256(urdf_path),
            "base_link": base_link,
            "tip_link": tip_link,
            "joint_names": list(chain.joint_names),
        },
        "action": {
            "shape": [len(ACTION_NAMES)],
            "names": list(ACTION_NAMES),
            "semantics": "RoboTwin-style EEF target relative to episode initial feedback pose",
            "translation": "target_xyz_base_link - initial_xyz_base_link",
            "rotation": "inverse(initial_quaternion) * target_quaternion, xyzw",
            "target_source": "FK of same-frame joint_absolute action",
            "gripper": "continuous normalized target: 0=minimum stroke, 1=maximum stroke",
            "gripper_physical_mapping": (
                f"0 -> {gripper_stroke_min_mm:g} mm, 1 -> {gripper_stroke_max_mm:g} mm"
            ),
            "gripper_stroke_min_mm": gripper_stroke_min_mm,
            "gripper_stroke_max_mm": gripper_stroke_max_mm,
            "lingbot_used_action_channel_ids": list(LINGBOT_EEF_ACTION_CHANNEL_IDS),
        },
        "cameras": {
            "ordered_keys": list(DEFAULT_RGB_IMAGE_KEYS),
            "layout": "width_concat",
        },
        "recommended_policy": {
            "checkpoint": "lerobot/lingbot_va_robotwin",
            "use_peft": True,
            "attn_mode": "flex",
            "obs_cam_keys": list(DEFAULT_RGB_IMAGE_KEYS),
            "camera_layout": "width_concat",
            "used_action_channel_ids": list(LINGBOT_EEF_ACTION_CHANNEL_IDS),
            "runtime_gripper_stroke_min_mm": gripper_stroke_min_mm,
            "runtime_gripper_stroke_max_mm": gripper_stroke_max_mm,
        },
        "validation": validation,
    }
    write_json(target_root / "meta/lingbot_va.json", manifest)
    (target_root / "TRAINING.md").write_text(
        _training_doc(gripper_stroke_min_mm, gripper_stroke_max_mm), encoding="utf-8"
    )
    package_sha256 = dataset_digest(target_root, exclude={Path("meta/lingbot_va.json")})
    manifest["package_sha256"] = package_sha256
    write_json(target_root / "meta/lingbot_va.json", manifest)

    if archive_path is not None:
        archive_path, archive_sha256 = write_tar_archive(
            target_root,
            archive_path=archive_path,
            root_name=final_target_root.name,
        )
        manifest["archive"] = str(archive_path)
        manifest["archive_sha256"] = archive_sha256
    return manifest


def _convert_data_files(
    *,
    data_files: list[Path],
    source_root: Path,
    target_root: Path,
    chain: SerialChainFK,
) -> _ConversionResult:
    actions: dict[int, np.ndarray] = {}
    states: dict[int, np.ndarray] = {}
    source_hash = hashlib.sha256()
    fk_position_errors: list[np.ndarray] = []
    fk_quaternion_dots: list[np.ndarray] = []
    roundtrip_position_errors: list[np.ndarray] = []
    roundtrip_quaternion_dots: list[np.ndarray] = []

    for path in data_files:
        frame = pd.read_parquet(path)
        source_hash.update((source_root / path.relative_to(target_root)).read_bytes())
        source_states = np.stack(frame["observation.state"].to_numpy()).astype(
            np.float64
        )
        source_actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        output_actions = np.empty((len(frame), len(ACTION_NAMES)), dtype=np.float32)
        output_states = source_states.astype(np.float32, copy=True)
        episode_indices = frame["episode_index"].to_numpy()

        for raw_index in frame["episode_index"].drop_duplicates().tolist():
            episode_index = int(raw_index)
            if episode_index in states:
                raise ValueError(
                    f"episode {episode_index} appears in more than one data file"
                )
            mask = episode_indices == raw_index
            episode = _convert_episode(
                episode_index=episode_index,
                observations=source_states[mask],
                joint_actions=source_actions[mask],
                chain=chain,
            )
            output_actions[mask] = episode["actions"]
            output_states[mask] = episode["states"]
            actions[episode_index] = episode["actions"]
            states[episode_index] = episode["states"]
            fk_position_errors.append(episode["fk_position_error"])
            fk_quaternion_dots.append(episode["fk_quaternion_dot"])
            roundtrip_position_errors.append(episode["roundtrip_position_error"])
            roundtrip_quaternion_dots.append(episode["roundtrip_quaternion_dot"])

        frame["observation.state"] = list(output_states)
        frame["action"] = list(output_actions)
        frame.to_parquet(path, index=False)

    return _ConversionResult(
        episode_actions=actions,
        episode_states=states,
        source_data_sha256=source_hash.hexdigest(),
        fk_position_errors=fk_position_errors,
        fk_quaternion_dots=fk_quaternion_dots,
        roundtrip_position_errors=roundtrip_position_errors,
        roundtrip_quaternion_dots=roundtrip_quaternion_dots,
    )


def _convert_episode(
    *,
    episode_index: int,
    observations: np.ndarray,
    joint_actions: np.ndarray,
    chain: SerialChainFK,
) -> dict[str, np.ndarray]:
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(joint_actions)):
        raise ValueError(f"episode {episode_index} contains non-finite vectors")
    arm_dof = len(JOINT_ACTION_NAMES) - 1
    eef_pose_dof = len(ACTION_NAMES) - 1
    initial_pose = observations[0, :eef_pose_dof]
    feedback_fk = np.stack(
        [
            chain.pose(values)
            for values in observations[:, eef_pose_dof : eef_pose_dof + arm_dof]
        ]
    )
    target_fk = np.stack([chain.pose(values) for values in joint_actions[:, :arm_dof]])
    _align_quaternion_signs(feedback_fk[:, 3:7], observations[:, 3:7])

    actions = np.empty((len(observations), len(ACTION_NAMES)), dtype=np.float64)
    for row, target_pose in enumerate(target_fk):
        actions[row, :7] = relative_pose(target_pose, initial_pose)
    _make_quaternions_continuous(actions[:, 3:7])
    actions[:, 7] = _normalized_gripper(
        joint_actions[:, arm_dof], episode_index=episode_index, label="action"
    )
    continuous_states = observations.astype(np.float32, copy=True)
    continuous_states[:, -1] = _normalized_gripper(
        observations[:, -1], episode_index=episode_index, label="state"
    )

    reconstructed = np.stack(
        [compose_relative_pose(values[:7], initial_pose) for values in actions]
    )
    _align_quaternion_signs(reconstructed[:, 3:7], target_fk[:, 3:7])
    return {
        "actions": actions.astype(np.float32),
        "states": continuous_states,
        "fk_position_error": np.linalg.norm(
            feedback_fk[:, :3] - observations[:, :3], axis=1
        ),
        "fk_quaternion_dot": np.abs(
            np.sum(feedback_fk[:, 3:7] * observations[:, 3:7], axis=1)
        ),
        "roundtrip_position_error": np.linalg.norm(
            reconstructed[:, :3] - target_fk[:, :3], axis=1
        ),
        "roundtrip_quaternion_dot": np.abs(
            np.sum(reconstructed[:, 3:7] * target_fk[:, 3:7], axis=1)
        ),
    }


def _normalized_gripper(
    values: np.ndarray, *, episode_index: int, label: str
) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"episode {episode_index} gripper {label} contains non-finite values"
        )
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(
            f"episode {episode_index} gripper {label} is outside normalized [0, 1]"
        )
    return np.clip(values, 0.0, 1.0)


def _validate_source(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError("source must be a LeRobot v3.0 dataset")
    features = info.get("features", {})
    if features.get("action", {}).get("names") != list(SOURCE_ACTION_NAMES):
        raise ValueError("source action must be A1 joint_absolute + normalized gripper")
    if features.get("observation.state", {}).get("names") != list(SOURCE_STATE_NAMES):
        raise ValueError(
            "source observation.state does not contain the expected A1 EEF and joints"
        )
    for key in DEFAULT_RGB_IMAGE_KEYS:
        if key not in features:
            raise ValueError(f"source is missing required camera feature {key!r}")


def _rewrite_info(target_root: Path, source_info: dict[str, Any]) -> None:
    info = json.loads(json.dumps(source_info))
    info["robot_type"] = "galaxea_a1_lingbot_eef"
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [len(ACTION_NAMES)],
        "names": list(ACTION_NAMES),
    }
    info["features"]["observation.state"]["names"][-1] = "gripper_normalized"
    write_json(target_root / "meta/info.json", info)


def _rewrite_global_stats(
    target_root: Path,
    action_stats: dict[str, list[float]],
    state_stats: dict[str, list[float]],
) -> None:
    path = target_root / "meta/stats.json"
    stats = read_json(path)
    stats["action"] = action_stats
    stats["observation.state"] = state_stats
    write_json(path, stats)


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


def _training_doc(gripper_stroke_min_mm: float, gripper_stroke_max_mm: float) -> str:
    return f"""# A1 LingBot-VA Dataset

The two ordered observations are the external `front` camera followed by the eye-in-hand
`wrist` camera. See `meta/lingbot_va.json` for the exact
EEF action convention, source hash, URDF hash, channel mapping, and validation results.

Use the RoboTwin checkpoint as the action-semantics baseline, LoRA/PEFT training, camera layout
`width_concat`, and action channels `[0,1,2,3,4,5,6,28]`. At inference, compose each predicted
EEF delta onto the episode's measured initial `base_link -> arm_seg6` pose before publishing the
absolute target. Gripper state and action are continuous normalized values. The hardware adapter
maps 0 to {gripper_stroke_min_mm:g} mm and 1 to {gripper_stroke_max_mm:g} mm linearly.
"""


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)
    config = load_pack_config(args.config)
    convert_raw_dataset(
        source_root=config.raw_source_root,
        target_root=config.base_v3_root,
        repo_id=config.base_v3_repo_id,
        overwrite=config.overwrite,
        expected_contract=config.source_contract,
    )
    base_v21_manifest = export_v21_dataset(
        source_root=config.base_v3_root,
        target_root=config.base_v21_target_root,
        repo_id=config.base_v21_repo_id,
        overwrite=config.overwrite,
        archive_path=config.base_v21_archive_path,
    )
    v3_manifest = pack_lingbot_dataset(
        source_root=config.base_v3_root,
        target_root=config.eef_v3_target_root,
        urdf_path=config.urdf_path,
        repo_id=config.eef_v3_repo_id,
        gripper_stroke_min_mm=config.gripper_stroke_min_mm,
        gripper_stroke_max_mm=config.gripper_stroke_max_mm,
        base_link=config.base_link,
        tip_link=config.tip_link,
        overwrite=config.overwrite,
        archive_path=config.eef_v3_archive_path,
    )
    v21_manifest = export_v21_dataset(
        source_root=config.eef_v3_target_root,
        target_root=config.eef_v21_target_root,
        repo_id=config.eef_v21_repo_id,
        overwrite=config.overwrite,
        archive_path=config.eef_v21_archive_path,
    )
    joint_v3_manifest = pack_joint_v3_dataset(
        source_root=config.base_v3_root,
        target_root=config.joint_v3_target_root,
        repo_id=config.joint_v3_repo_id,
        overwrite=config.overwrite,
        archive_path=config.joint_v3_archive_path,
    )
    print(
        json.dumps(
            {
                "base_v2.1": base_v21_manifest,
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
