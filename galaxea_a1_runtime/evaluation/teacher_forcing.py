"""Pure sequential Teacher Forcing action-comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from galaxea_a1_runtime.evaluation.metrics import summary
from galaxea_a1_runtime.evaluation.types import EpisodeRecord


def complete_block_starts(length: int, block_size: int) -> tuple[int, ...]:
    if length <= 0 or block_size <= 0 or length % block_size:
        raise ValueError(
            "sequential Teacher Forcing requires a positive episode length "
            "divisible by the model action block"
        )
    return tuple(range(0, length, block_size))


def action_step_records(
    *,
    model: str,
    episode: EpisodeRecord,
    action_start: int,
    prediction: np.ndarray,
    target: np.ndarray,
    inference_index: int,
) -> list[dict[str, Any]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 8
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
        or action_start < 0
        or action_start + prediction.shape[0] > episode.length
    ):
        raise ValueError(
            f"invalid sequential action comparison at {action_start}: "
            f"prediction={prediction.shape} target={target.shape}"
        )

    origin_xyz = episode.states[0, :3].astype(np.float64)
    records = []
    for offset, (predicted, actual) in enumerate(zip(prediction, target, strict=True)):
        action_index = action_start + offset
        current_relative_xyz = (
            episode.states[action_index, :3].astype(np.float64) - origin_xyz
        )
        predicted_norm = max(
            float(np.linalg.norm(predicted[3:7])), np.finfo(np.float64).eps
        )
        actual_norm = max(float(np.linalg.norm(actual[3:7])), np.finfo(np.float64).eps)
        quaternion_dot = abs(
            float(np.dot(predicted[3:7] / predicted_norm, actual[3:7] / actual_norm))
        )
        records.append(
            {
                "model": model,
                "episode_index": episode.episode_index,
                "inference_index": inference_index,
                "action_index": action_index,
                "timestamp_s": float(episode.timestamps[action_index]),
                "prediction": predicted.tolist(),
                "ground_truth_next_action": actual.tolist(),
                "xyz_error_vector_m": (predicted[:3] - actual[:3]).tolist(),
                "predicted_to_ground_truth_xyz_distance_m": float(
                    np.linalg.norm(predicted[:3] - actual[:3])
                ),
                "ground_truth_next_command_distance_m": float(
                    np.linalg.norm(actual[:3] - current_relative_xyz)
                ),
                "predicted_command_distance_from_ground_truth_state_m": float(
                    np.linalg.norm(predicted[:3] - current_relative_xyz)
                ),
                "gripper_abs_error": float(abs(predicted[7] - actual[7])),
                "quaternion_angle_deg": float(
                    np.degrees(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
                ),
            }
        )
    return records


def summarize_teacher_forcing(
    *,
    episode: EpisodeRecord,
    steps: list[dict[str, Any]],
    inferences: list[dict[str, Any]],
    unique_ground_truth_observation_steps: int,
    ground_truth_history_observation_inputs: int,
    ground_truth_action_input_steps: int,
    xyz_min: tuple[float, ...],
    xyz_max: tuple[float, ...],
    min_quat_norm: float,
) -> dict[str, Any]:
    if not steps or not inferences:
        raise ValueError("Teacher Forcing summary requires steps and inferences")

    def values(name: str) -> np.ndarray:
        return np.asarray([item[name] for item in steps], dtype=np.float64)

    xyz_error = values("predicted_to_ground_truth_xyz_distance_m")
    prediction = np.asarray([item["prediction"] for item in steps], dtype=np.float64)
    absolute_xyz = episode.states[0, :3].astype(np.float64) + prediction[:, :3]
    lo = np.asarray(xyz_min, dtype=np.float64)
    hi = np.asarray(xyz_max, dtype=np.float64)
    workspace_violation = np.any((absolute_xyz < lo) | (absolute_xyz > hi), axis=1)
    gripper_violation = (prediction[:, 7] < 0.0) | (prediction[:, 7] > 1.0)
    quaternion_norm = np.linalg.norm(prediction[:, 3:7], axis=1)
    worst_index = int(np.argmax(xyz_error))
    return {
        "predicted_action_steps": len(steps),
        "inference_calls": len(inferences),
        "unique_ground_truth_observation_steps": unique_ground_truth_observation_steps,
        "ground_truth_query_observation_inputs": len(inferences),
        "ground_truth_history_observation_inputs": (
            ground_truth_history_observation_inputs
        ),
        "ground_truth_action_input_steps": ground_truth_action_input_steps,
        "predicted_to_ground_truth_xyz_distance_m": summary(xyz_error),
        "ground_truth_next_command_distance_m": summary(
            values("ground_truth_next_command_distance_m")
        ),
        "predicted_command_distance_from_ground_truth_state_m": summary(
            values("predicted_command_distance_from_ground_truth_state_m")
        ),
        "gripper_abs_error": summary(values("gripper_abs_error")),
        "quaternion_angle_deg": summary(values("quaternion_angle_deg")),
        "inference_latency_s": summary(
            np.asarray([item["latency_s"] for item in inferences], dtype=np.float64)
        ),
        "prediction_gripper_range": [
            float(np.min(prediction[:, 7])),
            float(np.max(prediction[:, 7])),
        ],
        "prediction_quaternion_norm": summary(quaternion_norm),
        "raw_workspace_violation_steps": int(np.count_nonzero(workspace_violation)),
        "raw_gripper_violation_steps": int(np.count_nonzero(gripper_violation)),
        "quaternion_below_min_norm_steps": int(
            np.count_nonzero(quaternion_norm < min_quat_norm)
        ),
        "worst_xyz_step": steps[worst_index],
    }
