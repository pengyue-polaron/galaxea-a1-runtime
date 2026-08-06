"""Validate a canonical A1 LeRobot dataset without opening ROS or hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

from embodied_ops import print_dataset_report, standard_dataset_report

from galaxea_a1_runtime.apps.teleop.collection_task import normalize_collection_task
from galaxea_a1_runtime.apps.teleop.dataset_contract import direct_dataset_identity
from galaxea_a1_runtime.apps.teleop.metadata import collection_lifecycle_provenance
from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.console import ArgumentParser, failure
from galaxea_a1_runtime.lerobot.direct_recording import (
    validate_direct_dataset_provenance,
)
from galaxea_a1_runtime.teleop.config import (
    default_config_path,
    load_teleop_config,
    validate_collection_config,
)
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def dataset_report(
    config: TeleopConfig,
    *,
    experiment: str,
    task: str | None = None,
    allow_absent: bool = False,
) -> dict[str, object]:
    validate_collection_config(config)
    experiment = validate_experiment_name(experiment)
    identity = direct_dataset_identity(config, experiment)
    expected_task = normalize_collection_task(task) if task is not None else None
    state = validate_direct_dataset_provenance(
        identity,
        {"collection_lifecycle": collection_lifecycle_provenance(config)},
        expected_task=expected_task,
    )
    if state.total_episodes == 0 and not allow_absent:
        raise ValueError(f"canonical dataset does not exist: {identity.target_root}")
    return standard_dataset_report(
        robot="galaxea-a1",
        experiment=identity.experiment,
        root=str(identity.target_root),
        repo_id=identity.repo_id,
        episodes=state.total_episodes,
        frames=state.total_frames,
        tasks=state.tasks,
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-absent", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        root = args.repo_root.resolve()
        config_path = args.config or default_config_path(root)
        config = load_teleop_config(config_path, repo_root=root)
        report = dataset_report(
            config,
            experiment=args.experiment,
            task=args.task,
            allow_absent=args.allow_absent,
        )
        print_dataset_report(report, json_output=args.json)
    except (OSError, RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
