"""Sequential, hardware-free Teacher Forcing on one real EEF training episode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from embodied_ops.artifacts import read_json_object

from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata as lingbot_metadata
from galaxea_a1_runtime.apps.pi05.client import Pi05Client
from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
from galaxea_a1_runtime.apps.pi05.protocol import server_metadata as pi05_metadata
from galaxea_a1_runtime.console import ArgumentParser, info, step, success
from galaxea_a1_runtime.evaluation.eef_dataset import EefDataset
from galaxea_a1_runtime.evaluation.eef_policy_io import (
    camera_packet,
    lingbot_observation,
    pi05_observation,
    validated_lingbot_action,
    validated_pi05_actions,
)
from galaxea_a1_runtime.evaluation.eef_report import validate_training_provenance
from galaxea_a1_runtime.evaluation.io import (
    evaluation_run_dir,
    write_json_object,
    write_text_new,
)
from galaxea_a1_runtime.evaluation.offline_config import (
    OfflineEvalConfig,
    load_offline_eval_config,
)
from galaxea_a1_runtime.evaluation.teacher_forcing import (
    action_step_records,
    complete_block_starts,
    summarize_teacher_forcing,
)


REPORT_NAMES = {
    "lingbot": "lingbot_teacher_forcing.json",
    "pi05": "pi05_teacher_forcing.json",
}


def evaluate_lingbot_teacher_forcing(config: OfflineEvalConfig, run_id: str) -> Path:
    dataset = EefDataset(config)
    episode = dataset.episode(config.teacher_forcing_episode_index)
    deployment = load_lingbot_config(
        config.lingbot_deployment, repo_root=config.repo_root
    )
    policy = deployment.policy_server
    block_size = policy.action_per_frame
    block_starts = complete_block_starts(episode.length, block_size)
    if (
        deployment.execution.kv_observations_per_frame != block_size
        or policy.frame_chunk_size < 2
    ):
        raise ValueError(
            "LingBot sequential Teacher Forcing requires one GT observation per "
            "action and a conditioned first action frame"
        )
    images = dataset.frames(episode.episode_index, list(range(episode.length)))
    report = _base_report(
        model="lingbot",
        dataset=dataset,
        episode=episode,
        artifact_root=policy.model.artifact_root,
        semantics={
            "query_input": "ground-truth image and ground-truth EEF state",
            "history_input": (
                "four ground-truth post-action images and four ground-truth "
                "actions per temporal-cache update"
            ),
            "prediction_alignment": (
                "initial output action frame 1, then output action frame 0 after "
                "each fully teacher-forced cache block"
            ),
            "model_action_block_steps": block_size,
        },
    )
    validate_training_provenance(report, dataset, policy.model.artifact_root)
    client = LingBotClient(
        deployment.server.host,
        deployment.server.port,
        connect_timeout_s=deployment.server.connect_timeout_s,
        close_timeout_s=deployment.server.close_timeout_s,
        expected_metadata=lingbot_metadata(deployment),
    )
    steps: list[dict[str, Any]] = []
    inferences: list[dict[str, Any]] = []
    cache_updates: list[dict[str, Any]] = []
    last_action = None
    try:
        client.reset(episode.task)
        for inference_index, action_start in enumerate(block_starts):
            if action_start:
                previous_start = action_start - block_size
                target_history = episode.actions[previous_start:action_start]
                if inference_index == 1:
                    cache_state = last_action[:, :2, :].copy()
                    cache_state[:, 1, :] = target_history.T
                else:
                    cache_state = last_action[:, :1, :].copy()
                    cache_state[:, 0, :] = target_history.T
                observation_indices = list(range(previous_start + 1, action_start + 1))
                cache_started = time.monotonic()
                client.infer(
                    {
                        "obs": [
                            camera_packet(deployment, images[index])
                            for index in observation_indices
                        ],
                        "compute_kv_cache": True,
                        "imagine": False,
                        "state": cache_state,
                    }
                )
                cache_updates.append(
                    {
                        "cache_update_index": len(cache_updates),
                        "ground_truth_action_indices": [
                            previous_start,
                            action_start,
                        ],
                        "ground_truth_observation_indices": observation_indices,
                        "latency_s": time.monotonic() - cache_started,
                    }
                )

            infer_started = time.monotonic()
            action = validated_lingbot_action(
                client.infer(
                    lingbot_observation(
                        deployment, episode, images[action_start], action_start
                    )
                ),
                deployment,
            )
            latency = time.monotonic() - infer_started
            prediction_frame = 1 if inference_index == 0 else 0
            prediction = action[:, prediction_frame, :].T
            target = episode.actions[action_start : action_start + block_size]
            inferences.append(
                {
                    "inference_index": inference_index,
                    "ground_truth_query_observation_index": action_start,
                    "predicted_action_indices": [
                        action_start,
                        action_start + block_size,
                    ],
                    "latency_s": latency,
                }
            )
            steps.extend(
                action_step_records(
                    model="lingbot",
                    episode=episode,
                    action_start=action_start,
                    prediction=prediction,
                    target=target,
                    inference_index=inference_index,
                )
            )
            last_action = action
            if inference_index % 10 == 0 or inference_index + 1 == len(block_starts):
                info(
                    f"LingBot Teacher Forcing block {inference_index + 1}/"
                    f"{len(block_starts)} actions={action_start}:{action_start + block_size}"
                )
    finally:
        client.close()

    report["inferences"] = inferences
    report["cache_updates"] = cache_updates
    report["steps"] = steps
    report["summary"] = summarize_teacher_forcing(
        episode=episode,
        steps=steps,
        inferences=inferences,
        unique_ground_truth_observation_steps=episode.length - block_size + 1,
        ground_truth_history_observation_inputs=sum(
            len(item["ground_truth_observation_indices"]) for item in cache_updates
        ),
        ground_truth_action_input_steps=sum(
            end - start
            for start, end in (
                item["ground_truth_action_indices"] for item in cache_updates
            )
        ),
        xyz_min=deployment.system.eef.xyz_min,
        xyz_max=deployment.system.eef.xyz_max,
        min_quat_norm=deployment.system.eef.min_quat_norm,
    )
    report["after_first_ground_truth_cache_summary"] = summarize_teacher_forcing(
        episode=episode,
        steps=steps[block_size:],
        inferences=inferences[1:],
        unique_ground_truth_observation_steps=episode.length - block_size,
        ground_truth_history_observation_inputs=sum(
            len(item["ground_truth_observation_indices"]) for item in cache_updates
        ),
        ground_truth_action_input_steps=sum(
            end - start
            for start, end in (
                item["ground_truth_action_indices"] for item in cache_updates
            )
        ),
        xyz_min=deployment.system.eef.xyz_min,
        xyz_max=deployment.system.eef.xyz_max,
        min_quat_norm=deployment.system.eef.min_quat_norm,
    )
    return _write_report(config, run_id, "lingbot", report)


def evaluate_pi05_teacher_forcing(config: OfflineEvalConfig, run_id: str) -> Path:
    dataset = EefDataset(config)
    episode = dataset.episode(config.teacher_forcing_episode_index)
    deployment = load_pi05_config(config.pi05_deployment, repo_root=config.repo_root)
    images = dataset.frames(episode.episode_index, list(range(episode.length)))
    report = _base_report(
        model="pi05",
        dataset=dataset,
        episode=episode,
        artifact_root=deployment.model.artifact_root,
        semantics={
            "query_input": "ground-truth image and complete ground-truth state",
            "history_input": (
                "none; the pi0.5 service has no previous-action or temporal-cache input"
            ),
            "prediction_alignment": (
                "first action of the predicted horizon versus the same row's "
                "ground-truth next action"
            ),
            "model_action_block_steps": 1,
        },
    )
    validate_training_provenance(report, dataset, deployment.model.artifact_root)
    client = Pi05Client(
        deployment.server.host,
        deployment.server.port,
        connect_timeout_s=deployment.server.connect_timeout_s,
        close_timeout_s=deployment.server.close_timeout_s,
        expected_metadata=pi05_metadata(deployment),
    )
    steps: list[dict[str, Any]] = []
    inferences: list[dict[str, Any]] = []
    try:
        for frame_index in range(episode.length):
            started = time.monotonic()
            prediction = validated_pi05_actions(
                client.infer(
                    pi05_observation(
                        deployment, episode, images[frame_index], frame_index
                    )
                ),
                deployment,
            )
            latency = time.monotonic() - started
            inferences.append(
                {
                    "inference_index": frame_index,
                    "ground_truth_query_observation_index": frame_index,
                    "predicted_action_indices": [frame_index, frame_index + 1],
                    "latency_s": latency,
                }
            )
            steps.extend(
                action_step_records(
                    model="pi05",
                    episode=episode,
                    action_start=frame_index,
                    prediction=prediction[:1],
                    target=episode.actions[frame_index : frame_index + 1],
                    inference_index=frame_index,
                )
            )
            if frame_index % 50 == 0 or frame_index + 1 == episode.length:
                info(f"Pi0.5 Teacher Forcing step {frame_index + 1}/{episode.length}")
    finally:
        client.close()

    report["inferences"] = inferences
    report["cache_updates"] = []
    report["steps"] = steps
    report["summary"] = summarize_teacher_forcing(
        episode=episode,
        steps=steps,
        inferences=inferences,
        unique_ground_truth_observation_steps=episode.length,
        ground_truth_history_observation_inputs=0,
        ground_truth_action_input_steps=0,
        xyz_min=deployment.system.eef.xyz_min,
        xyz_max=deployment.system.eef.xyz_max,
        min_quat_norm=deployment.system.eef.min_quat_norm,
    )
    return _write_report(config, run_id, "pi05", report)


def summarize_run(config: OfflineEvalConfig, run_id: str) -> Path:
    run_dir = evaluation_run_dir(config, run_id)
    reports = {
        model: read_json_object(run_dir / name) for model, name in REPORT_NAMES.items()
    }
    lines = [
        "# Sequential Teacher Forcing",
        "",
        f"Run: `{run_id}`",
        "",
        "One complete real training episode is replayed in temporal order. Distances "
        "compare each predicted EEF target with the dataset's ground-truth next action.",
        "",
        "| Model | Episode steps | Inference calls | GT action history inputs | XYZ mean / P95 / max | Gripper MAE | Quaternion P95 | Latency median / P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, report in reports.items():
        summary = report["summary"]
        xyz = summary["predicted_to_ground_truth_xyz_distance_m"]
        gripper = summary["gripper_abs_error"]
        quaternion = summary["quaternion_angle_deg"]
        latency = summary["inference_latency_s"]
        lines.append(
            f"| {model} | {summary['predicted_action_steps']} | "
            f"{summary['inference_calls']} | "
            f"{summary['ground_truth_action_input_steps']} | "
            f"{xyz['mean']:.4f} / {xyz['p95']:.4f} / {xyz['max']:.4f} m | "
            f"{gripper['mean']:.4f} | {quaternion['p95']:.2f}° | "
            f"{latency['median']:.3f} / {latency['p95']:.3f} s |"
        )
    lines.extend(
        [
            "",
            "## First next-action comparison",
            "",
        ]
    )
    for model, report in reports.items():
        first = report["steps"][0]
        lines.append(
            f"- `{model}` step 0: XYZ distance="
            f"{first['predicted_to_ground_truth_xyz_distance_m'] * 1000.0:.2f} mm; "
            f"prediction={first['prediction']}; "
            f"ground_truth={first['ground_truth_next_action']}"
        )
    cached = reports["lingbot"]["after_first_ground_truth_cache_summary"]
    cached_xyz = cached["predicted_to_ground_truth_xyz_distance_m"]
    lines.extend(
        [
            "",
            "## LingBot after the first GT cache update",
            "",
            f"Across 444 cache-conditioned action steps, XYZ mean/P95/max is "
            f"{cached_xyz['mean'] * 1000.0:.2f}/"
            f"{cached_xyz['p95'] * 1000.0:.2f}/"
            f"{cached_xyz['max'] * 1000.0:.2f} mm.",
            "",
            "## Output safety audit",
            "",
        ]
    )
    for model, report in reports.items():
        summary = report["summary"]
        lines.append(
            f"- `{model}`: raw workspace violations="
            f"{summary['raw_workspace_violation_steps']}, raw gripper violations="
            f"{summary['raw_gripper_violation_steps']}, quaternion below minimum="
            f"{summary['quaternion_below_min_norm_steps']}"
        )
    lines.extend(
        [
            "",
            "## Semantics",
            "",
            "- LingBot predicts four control actions per model action frame. Before "
            "each subsequent inference, all four previous actions and their four "
            "post-action images are replaced with ground truth in the temporal cache.",
            "- Pi0.5 receives a ground-truth image and full ground-truth state at every "
            "dataset step. It has no action-history input channel, so its Teacher "
            "Forcing is observation/state replay rather than action-cache forcing.",
            "- This measures one-step behavior on training data; it does not measure "
            "autonomous error recovery or real-robot task success.",
            "",
        ]
    )
    path = run_dir / "TEACHER_FORCING_REPORT.md"
    write_text_new(path, "\n".join(lines))
    success(f"Teacher Forcing summary written: {path}")
    return path


def _base_report(
    *, model, dataset, episode, artifact_root, semantics
) -> dict[str, Any]:
    return {
        "schema_version": "galaxea_a1_eef_sequential_teacher_forcing_v1",
        "model": model,
        "created_unix_s": time.time(),
        "artifact_root": str(artifact_root),
        "dataset": {
            "repo_id": dataset.eef["repo_id"],
            "episodes": int(dataset.info["total_episodes"]),
            "frames": int(dataset.info["total_frames"]),
            "fps": int(dataset.info["fps"]),
        },
        "episode": {
            "episode_index": episode.episode_index,
            "task_index": episode.task_index,
            "task": episode.task,
            "steps": episode.length,
            "duration_s": float(episode.timestamps[-1] - episode.timestamps[0]),
        },
        "semantics": semantics,
        "limitations": [
            "the selected episode is part of the training set",
            "ground-truth replay prevents observation-distribution drift",
            "no ROS, cameras, serial devices, or robot hardware were opened",
        ],
    }


def _write_report(
    config: OfflineEvalConfig, run_id: str, model: str, report: dict[str, Any]
) -> Path:
    path = evaluation_run_dir(config, run_id) / REPORT_NAMES[model]
    write_json_object(path, report)
    success(f"{model} Teacher Forcing written: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lingbot", "pi05", "summarize"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = load_offline_eval_config(args.config, repo_root=root)
    step(f"Teacher Forcing command={args.command} run={args.run_id}")
    if args.command == "lingbot":
        evaluate_lingbot_teacher_forcing(config, args.run_id)
    elif args.command == "pi05":
        evaluate_pi05_teacher_forcing(config, args.run_id)
    else:
        summarize_run(config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
