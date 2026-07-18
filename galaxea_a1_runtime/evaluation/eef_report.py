"""Durable reports and checkpoint provenance for EEF policy evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.console import success
from galaxea_a1_runtime.evaluation.eef_dataset import EefDataset
from galaxea_a1_runtime.evaluation.io import (
    evaluation_run_dir,
    read_json_object,
    write_text_new,
)
from galaxea_a1_runtime.evaluation.offline_config import OfflineEvalConfig


MODEL_NAMES = ("lingbot", "pi05")


def base_report(
    model: str,
    config: OfflineEvalConfig,
    dataset: EefDataset,
    artifact_root: Path,
) -> dict[str, Any]:
    preflight = dataset.preflight()
    color_checks = {}
    for episode_indices in dataset.episodes_by_task().values():
        episode_index = episode_indices[0]
        color_checks[str(episode_index)] = dataset.color_alignment_check(episode_index)
    return {
        "schema_version": "galaxea_a1_eef_offline_eval_v1",
        "model": model,
        "created_unix_s": time.time(),
        "evaluation_config": str(config.path.relative_to(config.repo_root)),
        "artifact_root": str(artifact_root),
        "dataset": preflight,
        "color_and_trim_alignment": color_checks,
        "coverage": asdict(config.coverage),
        "limitations": [
            "evaluation data is the recorded training set",
            "teacher forcing does not test autonomous closed-loop recovery",
            "no ROS, camera, serial, or robot hardware was opened",
        ],
    }


def validate_training_provenance(
    report: dict[str, Any], dataset: EefDataset, artifact_root: Path
) -> None:
    training = read_json_object(artifact_root / "training_summary.json")
    expected = {
        "dataset": dataset.config.dataset_repo_id,
        "dataset_episodes": int(dataset.info["total_episodes"]),
        "dataset_frames": int(dataset.info["total_frames"]),
    }
    mismatched = {
        key: (training.get(key), value)
        for key, value in expected.items()
        if training.get(key) != value
    }
    if mismatched:
        raise ValueError(f"checkpoint/dataset provenance mismatch: {mismatched}")
    report["checkpoint_training_provenance"] = {
        **expected,
        "dataset_revision": training.get("dataset_revision"),
        "checkpoint_step": training.get("checkpoint_step"),
    }


def summarize_run(config: OfflineEvalConfig, run_id: str) -> Path:
    run_dir = evaluation_run_dir(config, run_id)
    reports = {name: read_json_object(run_dir / f"{name}.json") for name in MODEL_NAMES}
    lines = [
        "# Fruit-placement offline policy evaluation",
        "",
        f"Run: `{run_id}`",
        "",
        "This is a training-set regression evaluation: both checkpoints record the same "
        "130-episode dataset as their training source. It validates integration and "
        "training-distribution behavior, not held-out generalization.",
        "",
        "| Model | Cases | Action points | First-action XYZ mean / P95 | Horizon XYZ mean / P95 | Mean gripper error | P95 quaternion error | Mean latency | Raw gripper violations | Runtime-rejected steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in reports.items():
        summary = report["summary"]["overall"]
        lines.append(
            f"| {name} | {summary['cases']} | {summary['action_points']} | "
            f"{summary['first_action_xyz_error_m']['mean']:.4f} / "
            f"{summary['first_action_xyz_error_m']['p95']:.4f} m | "
            f"{summary['xyz_error_m']['mean']:.4f} / "
            f"{summary['xyz_error_m']['p95']:.4f} m | "
            f"{summary['gripper_abs_error']['mean']:.4f} | "
            f"{summary['quaternion_angle_deg']['p95']:.2f}° | "
            f"{summary['latency_s']['mean']:.3f} s | "
            f"{summary['raw_gripper_violation_steps']} | "
            f"{summary['runtime_rejected_steps']} |"
        )
    lines.extend(["", "## Scope breakdown", ""])
    for name, report in reports.items():
        lines.extend([f"### {name}", ""])
        for scope, summary in report["summary"]["by_scope"].items():
            lines.append(
                f"- `{scope}`: {summary['cases']} cases, "
                f"XYZ mean/p95={summary['xyz_error_m']['mean']:.4f}/"
                f"{summary['xyz_error_m']['p95']:.4f} m, "
                f"first-action mean/p95="
                f"{summary['first_action_xyz_error_m']['mean']:.4f}/"
                f"{summary['first_action_xyz_error_m']['p95']:.4f} m, "
                f"gripper MAE={summary['gripper_abs_error']['mean']:.4f}, "
                f"raw/target-workspace="
                f"{summary['raw_workspace_violation_steps']}/"
                f"{summary['target_workspace_violation_steps']}, "
                f"raw-gripper-violations={summary['raw_gripper_violation_steps']}, "
                f"runtime-rejected={summary['runtime_rejected_steps']}"
            )
        lines.extend(
            [
                "",
                f"![{name} first-frame contact sheet]({name}_contact_sheet.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Raw workspace and gripper excursions are reported; the runtime rejects them without publication.",
            "- A quaternion below the configured minimum norm is runtime-rejected rather than rewritten.",
            "- Demonstration workspace excursions are reported separately because some training targets already lie outside the current System workspace.",
            "- Teacher forcing prevents compounding state error and therefore does not measure autonomous task success.",
            "- A real hold-out split or newly collected episodes are required for generalization claims.",
            "",
        ]
    )
    path = run_dir / "REPORT.md"
    write_text_new(path, "\n".join(lines))
    success(f"Offline evaluation summary written: {path}")
    return path
