"""LeRobot-facing schema contracts for Galaxea A1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .constants import LEROBOT_DATASET_FORMAT

DEFAULT_STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)

EEF_DELTA_ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
)

EEF_TRANSLATION_ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "gripper",
)

JOINT_ACTION_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)


class ActionMode(StrEnum):
    EEF_DELTA = "eef_delta"
    EEF_TRANSLATION = "eef_translation"
    JOINT_ABSOLUTE = "joint_absolute"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    height: int
    width: int
    channels: int = 3

    def feature_key(self) -> str:
        return f"observation.images.{self.name}"

    def feature(self) -> dict[str, Any]:
        _validate_identifier(self.name, "camera name")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"invalid camera shape for {self.name}: {self.height}x{self.width}")
        if self.channels not in (1, 3, 4):
            raise ValueError(f"invalid channel count for {self.name}: {self.channels}")
        return {
            "dtype": "video",
            "shape": (self.height, self.width, self.channels),
            "names": ["height", "width", "channel"],
        }


@dataclass(frozen=True)
class DatasetContract:
    dataset_format: str
    action_mode: ActionMode
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_specs: tuple[CameraSpec, ...]

    def features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": vector_feature(self.state_names),
            "action": vector_feature(self.action_names),
        }
        for camera in self.camera_specs:
            features[camera.feature_key()] = camera.feature()
        return features


def default_dataset_contract(
    *,
    action_mode: ActionMode = ActionMode.EEF_DELTA,
    cameras: tuple[CameraSpec, ...] = (
        CameraSpec("front", height=480, width=640),
        CameraSpec("wrist", height=480, width=640),
    ),
) -> DatasetContract:
    return DatasetContract(
        dataset_format=LEROBOT_DATASET_FORMAT,
        action_mode=action_mode,
        state_names=DEFAULT_STATE_NAMES,
        action_names=action_names_for_mode(action_mode),
        camera_specs=cameras,
    )


def action_names_for_mode(action_mode: ActionMode) -> tuple[str, ...]:
    if action_mode == ActionMode.EEF_DELTA:
        return EEF_DELTA_ACTION_NAMES
    if action_mode == ActionMode.EEF_TRANSLATION:
        return EEF_TRANSLATION_ACTION_NAMES
    if action_mode == ActionMode.JOINT_ABSOLUTE:
        return JOINT_ACTION_NAMES
    raise ValueError(f"unsupported action mode: {action_mode}")


def vector_feature(names: tuple[str, ...]) -> dict[str, Any]:
    if not names:
        raise ValueError("vector feature must have at least one name")
    for name in names:
        _validate_identifier(name, "feature name")
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": list(names),
    }


def validate_frame_keys(
    frame: dict[str, Any],
    *,
    contract: DatasetContract,
) -> None:
    required = set(contract.features())
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"frame missing required keys: {missing}")


def _validate_identifier(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{label} must not contain whitespace: {value!r}")
