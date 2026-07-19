"""LeRobotDataset v3 creation helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import DatasetContract


class ImageStorage(StrEnum):
    """LeRobot image persistence mode.

    Production A1 collection always uses ``VIDEO``. ``IMAGE`` remains available
    only to small, hardware-free writer tests and generic conversion helpers.
    """

    VIDEO = "video"
    IMAGE = "image"


CANONICAL_IMAGE_STORAGE = ImageStorage.VIDEO
LEROBOT_GENERATED_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset writer settings; deliberately separate from physical system config."""

    repo_id: str
    root: Path
    fps: int
    robot_type: str = "galaxea_a1"
    image_storage: ImageStorage = CANONICAL_IMAGE_STORAGE

    def validate(self) -> None:
        validate_dataset_repo_id(self.repo_id)
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not isinstance(self.image_storage, ImageStorage):
            raise ValueError("image_storage must be an ImageStorage value")

    @property
    def use_videos(self) -> bool:
        return self.image_storage is ImageStorage.VIDEO


def validate_dataset_repo_id(value: str, *, label: str = "repo_id") -> None:
    """Validate the shared Hugging Face dataset identity contract."""

    if value.count("/") != 1:
        raise ValueError(f"{label} must be namespaced, for example 'user/a1_task'")
    try:
        validate_repo_id(value)
    except HFValidationError as exc:
        raise ValueError(f"invalid {label} {value!r}: {exc}") from exc


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
    features = deepcopy(contract.features())
    if config.image_storage is ImageStorage.IMAGE:
        for feature in features.values():
            if feature["dtype"] == "video":
                feature["dtype"] = "image"
    return {
        "repo_id": config.repo_id,
        "root": config.root,
        "fps": config.fps,
        "robot_type": config.robot_type,
        "features": features,
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


def resume_lerobot_dataset(*, repo_id: str, root: Path) -> Any:
    """Resume a local LeRobotDataset lazily for one atomic episode append."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.resume(repo_id=repo_id, root=root)
