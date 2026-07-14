"""Frame builders and writer wrapper for LeRobotDataset v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from galaxea_a1_runtime.hardware.io import A1Observation
from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import DatasetContract, validate_frame_keys

from .dataset import DatasetConfig, create_lerobot_dataset


def build_lerobot_frame(
    *,
    observation: A1Observation,
    action: RuntimeAction,
    task: str,
    contract: DatasetContract,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build one LeRobot frame from normalized runtime data."""

    if not task:
        raise ValueError("task must not be empty")
    observation.validate()
    if len(action.values) != len(contract.action_names):
        raise ValueError(
            f"action has {len(action.values)} values, need {len(contract.action_names)}"
        )

    frame: dict[str, Any] = {
        "observation.state": observation.state,
        "action": action.values,
        "task": task,
    }
    frame_timestamp = timestamp if timestamp is not None else observation.timestamp
    if frame_timestamp is not None:
        frame["timestamp"] = frame_timestamp
    for camera in contract.camera_specs:
        image_key = camera.name
        if image_key not in observation.images:
            raise ValueError(f"observation missing camera image: {image_key}")
        frame[camera.feature_key()] = observation.images[image_key]

    validate_frame_keys(frame, contract=contract)
    return frame


@dataclass
class LeRobotV3DatasetWriter:
    """Small wrapper around LeRobotDataset for runtime collection."""

    config: DatasetConfig
    contract: DatasetContract
    dataset: Any | None = None

    def open(self) -> None:
        if self.dataset is not None:
            return
        self.dataset = create_lerobot_dataset(config=self.config, contract=self.contract)

    def add_frame(
        self,
        *,
        observation: A1Observation,
        action: RuntimeAction,
        task: str,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("dataset writer is not open")
        frame = build_lerobot_frame(
            observation=observation,
            action=action,
            task=task,
            contract=self.contract,
            timestamp=timestamp,
        )
        return self.add_prebuilt_frame(frame)

    def add_prebuilt_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("dataset writer is not open")
        validate_frame_keys(frame, contract=self.contract)
        self.dataset.add_frame(frame)
        return frame

    def save_episode(self) -> None:
        if self.dataset is None:
            raise RuntimeError("dataset writer is not open")
        self.dataset.save_episode()

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()
