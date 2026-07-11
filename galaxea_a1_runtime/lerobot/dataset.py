"""LeRobotDataset v3 creation helpers."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.config import DatasetConfig
from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import DatasetContract


def build_dataset_create_kwargs(
    *,
    config: DatasetConfig,
    contract: DatasetContract,
) -> dict[str, Any]:
    """Build kwargs for `LeRobotDataset.create` without importing LeRobot."""

    config.validate()
    if contract.dataset_format != LEROBOT_DATASET_FORMAT:
        raise ValueError(
            f"unsupported dataset format: {contract.dataset_format}; "
            f"expected {LEROBOT_DATASET_FORMAT}"
        )
    return {
        "repo_id": config.repo_id,
        "root": config.root,
        "fps": config.fps,
        "robot_type": config.robot_type,
        "features": contract.features(),
        "use_videos": config.use_videos,
    }


def create_lerobot_dataset(
    *,
    config: DatasetConfig,
    contract: DatasetContract,
) -> Any:
    """Create a LeRobotDataset lazily so static tests do not require LeRobot."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(**build_dataset_create_kwargs(config=config, contract=contract))
