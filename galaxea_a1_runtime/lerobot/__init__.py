"""LeRobot integration helpers for Galaxea A1.

Imports are lazy so hardware-free CLI commands do not pull in torch or LeRobot
robot modules unless they are actually needed.
"""

from __future__ import annotations

__all__ = [
    "GalaxeaA1Robot",
    "GalaxeaA1RobotConfig",
    "LeRobotV3DatasetWriter",
    "LeRobotEpisodeRecorder",
    "MigrationPlan",
    "RecordedStep",
    "build_lerobot_frame",
    "build_dataset_create_kwargs",
    "create_lerobot_dataset",
    "dataset_contract_from_robot_config",
    "plan_raw_episodes_to_v30",
    "plan_v21_to_v30",
]


def __getattr__(name: str):
    if name in {"build_dataset_create_kwargs", "create_lerobot_dataset"}:
        from .dataset import build_dataset_create_kwargs, create_lerobot_dataset

        return {
            "build_dataset_create_kwargs": build_dataset_create_kwargs,
            "create_lerobot_dataset": create_lerobot_dataset,
        }[name]
    if name in {"MigrationPlan", "plan_raw_episodes_to_v30", "plan_v21_to_v30"}:
        from .migration import MigrationPlan, plan_raw_episodes_to_v30, plan_v21_to_v30

        return {
            "MigrationPlan": MigrationPlan,
            "plan_raw_episodes_to_v30": plan_raw_episodes_to_v30,
            "plan_v21_to_v30": plan_v21_to_v30,
        }[name]
    if name in {"GalaxeaA1Robot", "GalaxeaA1RobotConfig", "dataset_contract_from_robot_config"}:
        from .robot import GalaxeaA1Robot, GalaxeaA1RobotConfig, dataset_contract_from_robot_config

        return {
            "GalaxeaA1Robot": GalaxeaA1Robot,
            "GalaxeaA1RobotConfig": GalaxeaA1RobotConfig,
            "dataset_contract_from_robot_config": dataset_contract_from_robot_config,
        }[name]
    if name in {"LeRobotV3DatasetWriter", "build_lerobot_frame"}:
        from .writer import LeRobotV3DatasetWriter, build_lerobot_frame

        return {
            "LeRobotV3DatasetWriter": LeRobotV3DatasetWriter,
            "build_lerobot_frame": build_lerobot_frame,
        }[name]
    if name in {"LeRobotEpisodeRecorder", "RecordedStep"}:
        from .recorder import LeRobotEpisodeRecorder, RecordedStep

        return {
            "LeRobotEpisodeRecorder": LeRobotEpisodeRecorder,
            "RecordedStep": RecordedStep,
        }[name]
    raise AttributeError(name)
