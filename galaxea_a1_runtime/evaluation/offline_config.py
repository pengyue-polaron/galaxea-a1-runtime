"""Strict tracked configuration for EEF policy offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from galaxea_a1_runtime.configuration.base import (
    integer,
    load_toml,
    repo_path,
    require_exact_keys,
    required_table,
    string,
)


@dataclass(frozen=True)
class OfflineCoverage:
    lingbot_first_frame_episodes_per_task: int
    lingbot_teacher_forced_episodes_per_task: int
    lingbot_teacher_forced_chunks: int
    pi05_frames_per_episode: int


@dataclass(frozen=True)
class OfflineEvalConfig:
    path: Path
    repo_root: Path
    dataset_root: Path
    raw_root: Path
    dataset_repo_id: str
    lingbot_deployment: Path
    pi05_deployment: Path
    output_root: Path
    coverage: OfflineCoverage


def load_offline_eval_config(
    path: Path, *, repo_root: Path | None = None
) -> OfflineEvalConfig:
    path, root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"dataset", "deployments", "output", "coverage"},
        label="offline evaluation config",
    )
    dataset = required_table(data, "dataset")
    deployments = required_table(data, "deployments")
    output = required_table(data, "output")
    coverage = required_table(data, "coverage")
    require_exact_keys(
        dataset, required={"root", "raw_root", "repo_id"}, label="evaluation dataset"
    )
    require_exact_keys(
        deployments, required={"lingbot", "pi05"}, label="evaluation deployments"
    )
    require_exact_keys(output, required={"root"}, label="evaluation output")
    require_exact_keys(
        coverage,
        required={
            "lingbot_first_frame_episodes_per_task",
            "lingbot_teacher_forced_episodes_per_task",
            "lingbot_teacher_forced_chunks",
            "pi05_frames_per_episode",
        },
        label="evaluation coverage",
    )
    selection = OfflineCoverage(
        lingbot_first_frame_episodes_per_task=integer(
            coverage, "lingbot_first_frame_episodes_per_task"
        ),
        lingbot_teacher_forced_episodes_per_task=integer(
            coverage, "lingbot_teacher_forced_episodes_per_task"
        ),
        lingbot_teacher_forced_chunks=integer(
            coverage, "lingbot_teacher_forced_chunks"
        ),
        pi05_frames_per_episode=integer(coverage, "pi05_frames_per_episode"),
    )
    if any(value <= 0 for value in asdict(selection).values()):
        raise ValueError("offline evaluation coverage values must be positive")
    return OfflineEvalConfig(
        path=path,
        repo_root=root,
        dataset_root=repo_path(root, string(dataset, "root")),
        raw_root=repo_path(root, string(dataset, "raw_root")),
        dataset_repo_id=string(dataset, "repo_id"),
        lingbot_deployment=repo_path(root, string(deployments, "lingbot")),
        pi05_deployment=repo_path(root, string(deployments, "pi05")),
        output_root=repo_path(root, string(output, "root")),
        coverage=selection,
    )
