"""LeRobot-compatible Galaxea A1 robot adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.hardware.io import A1HardwareIO, NullA1HardwareIO
from galaxea_a1_runtime.policies.actions import normalize_action
from galaxea_a1_runtime.schema import (
    ActionMode,
    CameraSpec,
    DEFAULT_STATE_NAMES,
    DatasetContract,
    default_dataset_contract,
)

try:  # Keep core importable even when LeRobot is not installed.
    from lerobot.robots.config import RobotConfig as _LeRobotRobotConfig
    from lerobot.robots.robot import Robot as _LeRobotRobot
except Exception:  # pragma: no cover - exercised only in minimal environments.
    _LeRobotRobotConfig = None

    class _LeRobotRobot:  # type: ignore[no-redef]
        def __init__(self, config: Any):
            self.id = getattr(config, "id", None)
            self.calibration_dir = getattr(config, "calibration_dir", None)


if _LeRobotRobotConfig is None:

    @dataclass(kw_only=True)
    class _RobotConfigBase:
        id: str | None = None
        calibration_dir: Path | None = None

else:
    _RobotConfigBase = _LeRobotRobotConfig


@dataclass(kw_only=True)
class GalaxeaA1RobotConfig(_RobotConfigBase):
    id: str | None = "galaxea_a1"
    calibration_dir: Path | None = None
    action_mode: ActionMode = ActionMode.EEF_DELTA
    camera_specs: tuple[CameraSpec, ...] = field(
        default_factory=lambda: (
            CameraSpec("front", height=480, width=480),
            CameraSpec("wrist", height=480, width=640),
        )
    )


if _LeRobotRobotConfig is not None:
    GalaxeaA1RobotConfig = _LeRobotRobotConfig.register_subclass("galaxea_a1")(GalaxeaA1RobotConfig)


class GalaxeaA1Robot(_LeRobotRobot):
    """A LeRobot `Robot` wrapper around an A1 hardware IO adapter."""

    config_class = GalaxeaA1RobotConfig
    name = "galaxea_a1"

    def __init__(self, config: GalaxeaA1RobotConfig, io: A1HardwareIO | None = None):
        super().__init__(config)
        self.config = config
        self.io = io or NullA1HardwareIO()
        self.contract = default_dataset_contract(
            action_mode=config.action_mode,
            cameras=config.camera_specs,
        )

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        features: dict[str, type | tuple[int, ...]] = {
            "observation.state": (len(DEFAULT_STATE_NAMES),)
        }
        for camera in self.config.camera_specs:
            features[camera.feature_key()] = (camera.height, camera.width, camera.channels)
        return features

    @property
    def action_features(self) -> dict[str, type | tuple[int, ...]]:
        return {"action": (len(self.contract.action_names),)}

    @property
    def is_connected(self) -> bool:
        return self.io.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.io.connect()

    def configure(self) -> None:
        return None

    def calibrate(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        observation = self.io.get_observation()
        observation.validate()
        frame: dict[str, Any] = {"observation.state": observation.state}
        for name, image in observation.images.items():
            frame[f"observation.images.{name}"] = image
        if observation.timestamp is not None:
            frame["timestamp"] = observation.timestamp
        return frame

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        raw = action["action"] if "action" in action else action
        runtime_action = normalize_action(
            raw,
            mode=self.config.action_mode,
        )
        sent = self.io.send_runtime_action(runtime_action)
        return sent.as_dict()

    def disconnect(self) -> None:
        self.io.disconnect()


def dataset_contract_from_robot_config(config: GalaxeaA1RobotConfig) -> DatasetContract:
    return default_dataset_contract(action_mode=config.action_mode, cameras=config.camera_specs)
