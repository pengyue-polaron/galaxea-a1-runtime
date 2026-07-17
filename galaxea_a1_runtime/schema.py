"""LeRobot-facing schema contracts for Galaxea A1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from galaxea_a1_runtime.configuration.cameras import SystemRealSenseCameraConfig

from .constants import LEROBOT_DATASET_FORMAT

if TYPE_CHECKING:
    from galaxea_a1_runtime.configuration.system import SystemConfig

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

JOINT_ACTION_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)

JOINT_ACTION_NAMES_RAD = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_normalized",
)

EEF_ACTION_NAMES = (
    "eef_delta_x_from_episode_start",
    "eef_delta_y_from_episode_start",
    "eef_delta_z_from_episode_start",
    "eef_delta_qx_from_episode_start",
    "eef_delta_qy_from_episode_start",
    "eef_delta_qz_from_episode_start",
    "eef_delta_qw_from_episode_start",
    "gripper_normalized",
)

EEF_DATASET_STATE_NAMES = (*DEFAULT_STATE_NAMES[:7], *JOINT_ACTION_NAMES_RAD)

FRONT_IMAGE_KEY = "observation.images.front"
WRIST_IMAGE_KEY = "observation.images.wrist"
LINGBOT_EEF_ACTION_CHANNEL_IDS = (0, 1, 2, 3, 4, 5, 6, 28)
DEFAULT_RGB_IMAGE_KEYS = (FRONT_IMAGE_KEY, WRIST_IMAGE_KEY)


class ActionMode(StrEnum):
    JOINT_ABSOLUTE = "joint_absolute"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    height: int
    width: int
    channels: int = 3
    is_depth_map: bool = False
    depth_unit: str | None = None

    def feature_key(self) -> str:
        return f"observation.images.{self.name}"

    def feature(self) -> dict[str, Any]:
        _validate_identifier(self.name, "camera name")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"invalid camera shape for {self.name}: {self.height}x{self.width}"
            )
        if self.channels not in (1, 3, 4):
            raise ValueError(f"invalid channel count for {self.name}: {self.channels}")
        feature = {
            "dtype": "video",
            "shape": (self.height, self.width, self.channels),
            "names": ["height", "width", "channel"],
        }
        if self.is_depth_map:
            info: dict[str, Any] = {"is_depth_map": True}
            if self.depth_unit is not None:
                info["depth_unit"] = self.depth_unit
            feature["info"] = info
        return feature


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
    cameras: tuple[CameraSpec, ...],
) -> DatasetContract:
    return DatasetContract(
        dataset_format=LEROBOT_DATASET_FORMAT,
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=DEFAULT_STATE_NAMES,
        action_names=JOINT_ACTION_NAMES,
        camera_specs=cameras,
    )


def camera_specs_from_system(
    system: SystemConfig, *, include_depth: bool | None = None
) -> tuple[CameraSpec, ...]:
    """Derive dataset image shapes from the unique physical camera config."""

    front = system.cameras.front
    wrist = system.cameras.wrist
    front_width = front.crop.width if front.crop is not None else front.width
    front_height = front.crop.height if front.crop is not None else front.height
    specs = [
        CameraSpec("front", height=front_height, width=front_width),
        CameraSpec("wrist", height=wrist.height, width=wrist.width),
    ]
    depth_enabled = (
        isinstance(front, SystemRealSenseCameraConfig) and front.depth
        if include_depth is None
        else include_depth
    )
    if depth_enabled:
        if not isinstance(front, SystemRealSenseCameraConfig) or not front.depth:
            raise ValueError("front depth camera spec requested but depth is disabled")
        if front.crop is not None:
            depth_width, depth_height = front.crop.width, front.crop.height
        elif front.align_depth_to_color:
            depth_width, depth_height = front.width, front.height
        else:
            if front.depth_width is None or front.depth_height is None:
                raise ValueError("front depth dimensions are missing")
            depth_width, depth_height = front.depth_width, front.depth_height
        specs.append(
            CameraSpec(
                "front_depth",
                height=depth_height,
                width=depth_width,
                channels=1,
                is_depth_map=True,
                depth_unit="millimeter",
            )
        )
    return tuple(specs)


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
