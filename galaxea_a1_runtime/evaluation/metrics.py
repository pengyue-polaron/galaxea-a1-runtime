"""Pure EEF policy regression and output-safety metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from galaxea_a1_runtime.evaluation.types import EpisodeRecord


def case_metrics(
    *,
    model: str,
    scope: str,
    episode: EpisodeRecord,
    frame_index: int,
    prediction: np.ndarray,
    target: np.ndarray,
    latency_s: float,
    xyz_min,
    xyz_max,
    min_quat_norm: float,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 8
    ):
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape}, {target.shape}"
        )
    xyz_error = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    gripper_error = np.abs(prediction[:, 7] - target[:, 7])
    pred_norm = np.linalg.norm(prediction[:, 3:7], axis=1)
    target_norm = np.linalg.norm(target[:, 3:7], axis=1)
    safe_pred_norm = np.maximum(pred_norm, np.finfo(np.float64).eps)
    safe_target_norm = np.maximum(target_norm, np.finfo(np.float64).eps)
    dots = np.abs(
        np.sum(
            prediction[:, 3:7]
            / safe_pred_norm[:, None]
            * (target[:, 3:7] / safe_target_norm[:, None]),
            axis=1,
        )
    )
    quat_angle = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    origin_xyz = episode.states[0, :3].astype(np.float64)
    absolute_xyz = origin_xyz + prediction[:, :3]
    lo = np.asarray(xyz_min, dtype=np.float64)
    hi = np.asarray(xyz_max, dtype=np.float64)
    clamped = np.clip(absolute_xyz, lo, hi)
    clamp_distance = np.linalg.norm(clamped - absolute_xyz, axis=1)
    workspace_violation = np.any((absolute_xyz < lo) | (absolute_xyz > hi), axis=1)
    target_absolute_xyz = origin_xyz + target[:, :3]
    target_clamped = np.clip(target_absolute_xyz, lo, hi)
    target_clamp_distance = np.linalg.norm(target_clamped - target_absolute_xyz, axis=1)
    target_workspace_violation = np.any(
        (target_absolute_xyz < lo) | (target_absolute_xyz > hi), axis=1
    )
    gripper_violation = (prediction[:, 7] < 0.0) | (prediction[:, 7] > 1.0)
    quat_invalid = pred_norm < min_quat_norm
    output_rewrite = workspace_violation | gripper_violation
    return {
        "model": model,
        "scope": scope,
        "episode_index": episode.episode_index,
        "task_index": episode.task_index,
        "task": episode.task,
        "frame_index": frame_index,
        "action_points": int(prediction.shape[0]),
        "latency_s": float(latency_s),
        "xyz_error_m": summary(xyz_error),
        "gripper_abs_error": summary(gripper_error),
        "quaternion_angle_deg": summary(quat_angle),
        "prediction_quaternion_norm": summary(pred_norm),
        "xyz_error_values_m": xyz_error.tolist(),
        "gripper_abs_error_values": gripper_error.tolist(),
        "quaternion_angle_values_deg": quat_angle.tolist(),
        "raw_workspace_violation_steps": int(np.count_nonzero(workspace_violation)),
        "workspace_clamp_distance_m": summary(clamp_distance),
        "target_workspace_violation_steps": int(
            np.count_nonzero(target_workspace_violation)
        ),
        "target_workspace_clamp_distance_m": summary(target_clamp_distance),
        "raw_gripper_violation_steps": int(np.count_nonzero(gripper_violation)),
        "raw_output_rewrite_steps": int(np.count_nonzero(output_rewrite)),
        "quaternion_below_min_norm_steps": int(np.count_nonzero(quat_invalid)),
        "runtime_rejected_steps": int(np.count_nonzero(quat_invalid)),
        "prediction_first": prediction[0].tolist(),
        "target_first": target[0].tolist(),
        "prediction_last": prediction[-1].tolist(),
        "target_last": target[-1].tolist(),
    }


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("offline evaluation produced no cases")

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        def point_metric(name: str) -> dict[str, float]:
            values = np.concatenate(
                [np.asarray(item[name], dtype=np.float64) for item in selected]
            )
            return summary(values)

        prediction_first = np.asarray(
            [item["prediction_first"] for item in selected], dtype=np.float64
        )
        target_first = np.asarray(
            [item["target_first"] for item in selected], dtype=np.float64
        )
        first_xyz_error = np.linalg.norm(
            prediction_first[:, :3] - target_first[:, :3], axis=1
        )
        return {
            "cases": len(selected),
            "episodes": len({item["episode_index"] for item in selected}),
            "action_points": int(sum(item["action_points"] for item in selected)),
            "xyz_error_m": point_metric("xyz_error_values_m"),
            "first_action_xyz_error_m": summary(first_xyz_error),
            "gripper_abs_error": point_metric("gripper_abs_error_values"),
            "quaternion_angle_deg": point_metric("quaternion_angle_values_deg"),
            "latency_s": summary(
                np.asarray([item["latency_s"] for item in selected], dtype=np.float64)
            ),
            "raw_workspace_violation_steps": int(
                sum(item["raw_workspace_violation_steps"] for item in selected)
            ),
            "target_workspace_violation_steps": int(
                sum(item["target_workspace_violation_steps"] for item in selected)
            ),
            "raw_gripper_violation_steps": int(
                sum(item["raw_gripper_violation_steps"] for item in selected)
            ),
            "raw_output_rewrite_steps": int(
                sum(item["raw_output_rewrite_steps"] for item in selected)
            ),
            "runtime_rejected_steps": int(
                sum(item["runtime_rejected_steps"] for item in selected)
            ),
        }

    scopes = sorted({item["scope"] for item in cases})
    tasks = sorted({item["task"] for item in cases})
    return {
        "overall": aggregate(cases),
        "by_scope": {
            scope: aggregate([item for item in cases if item["scope"] == scope])
            for scope in scopes
        },
        "by_task": {
            task: aggregate([item for item in cases if item["task"] == task])
            for task in tasks
        },
    }


def compare_stats(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result = {}
    names = sorted(set(actual) & set(expected) & {"mean", "std", "q01", "q99"})
    for name in names:
        left = np.asarray(actual[name], dtype=np.float64)
        right = np.asarray(expected[name], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"normalization statistic shape mismatch for {name}")
        result[name] = {
            "max_abs_difference": float(np.max(np.abs(left - right))),
            "actual": left.tolist(),
            "checkpoint": right.tolist(),
        }
    return result


def vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary requires non-empty finite values")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def even_indices(max_start: int, count: int) -> list[int]:
    if max_start < 0:
        raise ValueError("episode is shorter than the action horizon")
    return sorted(set(np.linspace(0, max_start, num=count, dtype=np.int64).tolist()))
