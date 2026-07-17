"""Real-dataset, hardware-free evaluation for A1 EEF policy services."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

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
from galaxea_a1_runtime.evaluation.eef_report import (
    MODEL_NAMES,
    base_report,
    summarize_run,
    validate_training_provenance,
)
from galaxea_a1_runtime.evaluation.io import (
    evaluation_run_dir,
    read_json_object,
    write_contact_sheet,
    write_json_object,
)
from galaxea_a1_runtime.evaluation.metrics import (
    aggregate_cases,
    case_metrics,
    compare_stats,
    even_indices,
)
from galaxea_a1_runtime.evaluation.offline_config import (
    OfflineEvalConfig,
    load_offline_eval_config,
)


def evaluate_lingbot(config: OfflineEvalConfig, run_id: str) -> Path:
    dataset = EefDataset(config)
    deployment = load_lingbot_config(
        config.lingbot_deployment, repo_root=config.repo_root
    )
    artifact_root = deployment.policy_server.model.artifact_root
    report = base_report("lingbot", config, dataset, artifact_root)
    validate_training_provenance(report, dataset, artifact_root)
    report["normalization_comparison"] = compare_stats(
        report["dataset"]["action_stats"],
        {
            "q01": list(deployment.policy_server.q01_source),
            "q99": list(deployment.policy_server.q99_source),
        },
    )
    client = LingBotClient(
        deployment.server.host,
        deployment.server.port,
        connect_timeout_s=deployment.server.connect_timeout_s,
        close_timeout_s=deployment.server.close_timeout_s,
        expected_metadata=lingbot_metadata(deployment),
    )
    cases: list[dict] = []
    visuals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    visual_task_ids: set[int] = set()
    task_episodes = dataset.episodes_by_task()
    try:
        for task_index, episode_indices in sorted(task_episodes.items()):
            selected = episode_indices[
                : config.coverage.lingbot_first_frame_episodes_per_task
            ]
            if len(selected) < config.coverage.lingbot_first_frame_episodes_per_task:
                raise ValueError(f"not enough LingBot episodes for task {task_index}")
            teacher = set(
                selected[: config.coverage.lingbot_teacher_forced_episodes_per_task]
            )
            for episode_index in selected:
                episode = dataset.episode(episode_index)
                max_teacher_frame = (
                    config.coverage.lingbot_teacher_forced_chunks * 4
                    if episode_index in teacher
                    else 0
                )
                indices = (
                    list(range(max_teacher_frame + 1)) if max_teacher_frame else [0]
                )
                images = dataset.frames(episode_index, indices)
                client.reset(episode.task)
                first_started = time.monotonic()
                first = validated_lingbot_action(
                    client.infer(
                        lingbot_observation(deployment, episode, images[0], 0)
                    ),
                    deployment,
                )
                latency = time.monotonic() - first_started
                prediction = first[:, 1:, :].transpose(1, 2, 0).reshape(-1, 8)
                target = episode.actions[: prediction.shape[0]]
                cases.append(
                    case_metrics(
                        model="lingbot",
                        scope="first_frame_open_loop",
                        episode=episode,
                        frame_index=0,
                        prediction=prediction,
                        target=target,
                        latency_s=latency,
                        xyz_min=deployment.system.eef.xyz_min,
                        xyz_max=deployment.system.eef.xyz_max,
                        min_quat_norm=deployment.system.eef.min_quat_norm,
                    )
                )
                if task_index not in visual_task_ids:
                    visual_task_ids.add(task_index)
                    visuals.append(
                        (
                            episode.task,
                            images[0]["observation.images.front"],
                            images[0]["observation.images.wrist"],
                            prediction[0],
                            target[0],
                        )
                    )
                if episode_index in teacher:
                    cases.extend(
                        _lingbot_teacher_forced_cases(
                            client=client,
                            deployment=deployment,
                            episode=episode,
                            images=images,
                            first=first,
                            chunks=config.coverage.lingbot_teacher_forced_chunks,
                        )
                    )
                info(
                    f"LingBot offline episode {episode_index + 1}/"
                    f"{len(dataset.episode_meta)} task={episode.task!r}"
                )
    finally:
        client.close()
    report["cases"] = cases
    report["summary"] = aggregate_cases(cases)
    run_dir = evaluation_run_dir(config, run_id)
    write_contact_sheet(run_dir / "lingbot_contact_sheet.png", "LingBot", visuals)
    path = run_dir / "lingbot.json"
    write_json_object(path, report)
    success(f"LingBot offline evaluation written: {path}")
    return path


def _lingbot_teacher_forced_cases(
    *, client, deployment, episode, images, first, chunks: int
) -> list[dict]:
    cases: list[dict] = []
    target = episode.actions[:4]
    cache_state = first[:, :2, :].copy()
    cache_state[:, 1, :] = target.T
    client.infer(
        {
            "obs": [camera_packet(deployment, images[index]) for index in range(1, 5)],
            "compute_kv_cache": True,
            "imagine": False,
            "state": cache_state,
        }
    )
    for chunk_index in range(1, chunks):
        frame_index = chunk_index * 4
        started = time.monotonic()
        action = validated_lingbot_action(
            client.infer(
                lingbot_observation(
                    deployment, episode, images[frame_index], frame_index
                )
            ),
            deployment,
        )
        latency = time.monotonic() - started
        prediction = action[:, 0, :].T
        target = episode.actions[frame_index : frame_index + 4]
        cases.append(
            case_metrics(
                model="lingbot",
                scope="teacher_forced_cache_replay",
                episode=episode,
                frame_index=frame_index,
                prediction=prediction,
                target=target,
                latency_s=latency,
                xyz_min=deployment.system.eef.xyz_min,
                xyz_max=deployment.system.eef.xyz_max,
                min_quat_norm=deployment.system.eef.min_quat_norm,
            )
        )
        cache_state = action[:, :1, :].copy()
        cache_state[:, 0, :] = target.T
        client.infer(
            {
                "obs": [
                    camera_packet(deployment, images[index])
                    for index in range(frame_index + 1, frame_index + 5)
                ],
                "compute_kv_cache": True,
                "imagine": False,
                "state": cache_state,
            }
        )
    return cases


def evaluate_pi05(config: OfflineEvalConfig, run_id: str) -> Path:
    dataset = EefDataset(config)
    deployment = load_pi05_config(config.pi05_deployment, repo_root=config.repo_root)
    artifact_root = deployment.model.artifact_root
    report = base_report("pi05", config, dataset, artifact_root)
    validate_training_provenance(report, dataset, artifact_root)
    norm_stats = read_json_object(deployment.model_contract.norm_stats_path)[
        "norm_stats"
    ]
    report["normalization_comparison"] = {
        "state": compare_stats(report["dataset"]["state_stats"], norm_stats["state"]),
        "actions": compare_stats(
            report["dataset"]["action_stats"], norm_stats["actions"]
        ),
        "note": (
            "OpenPI training statistics are computed over horizon-sampled actions; "
            "small differences from one-copy-per-frame package statistics are expected."
        ),
    }
    client = Pi05Client(
        deployment.server.host,
        deployment.server.port,
        connect_timeout_s=deployment.server.connect_timeout_s,
        close_timeout_s=deployment.server.close_timeout_s,
        expected_metadata=pi05_metadata(deployment),
    )
    cases: list[dict] = []
    visuals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    visual_task_ids: set[int] = set()
    try:
        for episode_index in range(len(dataset.episode_meta)):
            episode = dataset.episode(episode_index)
            max_start = episode.length - deployment.model_contract.action_horizon
            indices = even_indices(max_start, config.coverage.pi05_frames_per_episode)
            images = dataset.frames(episode_index, indices)
            for frame_index in indices:
                packet = pi05_observation(
                    deployment, episode, images[frame_index], frame_index
                )
                started = time.monotonic()
                response = client.infer(packet)
                latency = time.monotonic() - started
                prediction = validated_pi05_actions(response, deployment)
                target = episode.actions[
                    frame_index : frame_index + deployment.model_contract.action_horizon
                ]
                scope = (
                    "first_frame"
                    if frame_index == 0
                    else "teacher_forced_observation_replay"
                )
                cases.append(
                    case_metrics(
                        model="pi05",
                        scope=scope,
                        episode=episode,
                        frame_index=frame_index,
                        prediction=prediction,
                        target=target,
                        latency_s=latency,
                        xyz_min=deployment.system.eef.xyz_min,
                        xyz_max=deployment.system.eef.xyz_max,
                        min_quat_norm=deployment.system.eef.min_quat_norm,
                    )
                )
                if frame_index == 0 and episode.task_index not in visual_task_ids:
                    visual_task_ids.add(episode.task_index)
                    visuals.append(
                        (
                            episode.task,
                            images[0]["observation.images.front"],
                            images[0]["observation.images.wrist"],
                            prediction[0],
                            target[0],
                        )
                    )
            info(
                f"Pi0.5 offline episode {episode_index + 1}/"
                f"{len(dataset.episode_meta)} task={episode.task!r}"
            )
    finally:
        client.close()
    report["cases"] = cases
    report["summary"] = aggregate_cases(cases)
    run_dir = evaluation_run_dir(config, run_id)
    write_contact_sheet(run_dir / "pi05_contact_sheet.png", "Pi0.5", visuals)
    path = run_dir / "pi05.json"
    write_json_object(path, report)
    success(f"Pi0.5 offline evaluation written: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*MODEL_NAMES, "summarize"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = load_offline_eval_config(args.config, repo_root=root)
    step(f"Offline evaluation command={args.command} run={args.run_id}")
    if args.command == "lingbot":
        evaluate_lingbot(config, args.run_id)
    elif args.command == "pi05":
        evaluate_pi05(config, args.run_id)
    else:
        summarize_run(config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
