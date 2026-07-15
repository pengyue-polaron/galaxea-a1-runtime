"""LeRobotDataset v3 creation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import DatasetContract


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset writer settings; deliberately separate from physical system config."""

    repo_id: str
    root: Path
    fps: int
    robot_type: str = "galaxea_a1"
    use_videos: bool = True

    def validate(self) -> None:
        if "/" not in self.repo_id:
            raise ValueError("repo_id should be namespaced, for example 'user/a1_task'")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")


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

    return LeRobotDataset.create(
        **build_dataset_create_kwargs(config=config, contract=contract)
    )
