"""LeRobot-facing schema contracts for Galaxea A1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from galaxea_a1_runtime.configuration.cameras import SystemRealSenseCameraConfig

if TYPE_CHECKING:
    from galaxea_a1_runtime.configuration.system import SystemConfig

EEF_POSE_STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
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

A1_STATE_NAMES = (*EEF_POSE_STATE_NAMES, *JOINT_ACTION_NAMES_RAD)

LINGBOT_EEF_ACTION_CHANNEL_IDS = (0, 1, 2, 3, 4, 5, 6, 28)
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"
TASK_FEATURE_KEY = "task"
IMAGE_FEATURE_PREFIX = "observation.images."
FRONT_IMAGE_FEATURE_KEY = f"{IMAGE_FEATURE_PREFIX}front"
WRIST_IMAGE_FEATURE_KEY = f"{IMAGE_FEATURE_PREFIX}wrist"
FRONT_DEPTH_FEATURE_KEY = f"{IMAGE_FEATURE_PREFIX}front_depth"
DEFAULT_RGB_IMAGE_KEYS = (
    FRONT_IMAGE_FEATURE_KEY,
    WRIST_IMAGE_FEATURE_KEY,
)
DIRECT_DATASET_SCHEMA_VERSION = "galaxea_a1_lerobot_dataset_v3_v2"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    height: int
    width: int
    channels: int = 3
    is_depth_map: bool = False
    depth_unit: str | None = None

    def feature_key(self) -> str:
        return f"{IMAGE_FEATURE_PREFIX}{self.name}"

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
            "names": ["height", "width", "channels"],
        }
        if self.is_depth_map:
            info: dict[str, Any] = {"is_depth_map": True}
            if self.depth_unit is not None:
                info["depth_unit"] = self.depth_unit
            feature["info"] = info
        return feature


@dataclass(frozen=True)
class DatasetContract:
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_specs: tuple[CameraSpec, ...]

    def features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            STATE_FEATURE_KEY: vector_feature(self.state_names),
            ACTION_FEATURE_KEY: vector_feature(self.action_names),
        }
        for camera in self.camera_specs:
            features[camera.feature_key()] = camera.feature()
        return features


def canonical_dataset_contract(
    *,
    cameras: tuple[CameraSpec, ...],
) -> DatasetContract:
    """Return the directly recorded, model-agnostic A1 LeRobot contract."""

    return DatasetContract(
        state_names=A1_STATE_NAMES,
        action_names=JOINT_ACTION_NAMES_RAD,
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


def _validate_identifier(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{label} must not contain whitespace: {value!r}")
