"""Validate a canonical A1 LeRobot dataset without opening ROS or hardware."""

from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.apps.teleop.dataset_contract import direct_dataset_identity
from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.lerobot.direct_recording import inspect_direct_dataset
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    config_path = args.config or default_config_path(root)
    config = load_teleop_config(config_path, repo_root=root)
    experiment = validate_experiment_name(args.experiment)
    identity = direct_dataset_identity(config, experiment)
    state = inspect_direct_dataset(identity)
    print(
        json.dumps(
            {
                "root": str(identity.target_root),
                "repo_id": identity.repo_id,
                "experiment": identity.experiment,
                "task": state.task,
                "total_episodes": state.total_episodes,
                "total_frames": state.total_frames,
                "status": "absent" if state.total_episodes == 0 else "valid",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
