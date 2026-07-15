"""Typed physical camera configuration shared by every A1 application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    require_exact_keys,
    required_table,
    string,
    text,
)
from galaxea_a1_runtime.configuration.image import ImageRoi, parse_optional_image_roi


@dataclass(frozen=True)
class SystemCameraDeviceConfig:
    backend: str
    width: int
    height: int
    fps: int
    crop: ImageRoi | None


@dataclass(frozen=True)
class SystemRealSenseCameraConfig(SystemCameraDeviceConfig):
    serial: str
    require_usb3: bool
    depth: bool
    depth_width: int | None
    depth_height: int | None
    align_depth_to_color: bool | None
    auto_exposure: bool
    exposure: int | None
    gain: int | None
    auto_white_balance: bool
    white_balance: int | None


@dataclass(frozen=True)
class SystemV4l2CameraConfig(SystemCameraDeviceConfig):
    device: str
    backend_api: str
    pixel_format: str


@dataclass(frozen=True)
class SystemCamerasConfig:
    warmup_frames: int
    max_age_s: float
    max_pair_skew_s: float
    front: SystemCameraDeviceConfig
    wrist: SystemCameraDeviceConfig


def parse_system_cameras(data: dict[str, Any]) -> SystemCamerasConfig:
    """Parse and validate the complete physical dual-camera contract."""

    require_exact_keys(
        data,
        required={"warmup_frames", "max_age_s", "max_pair_skew_s", "front", "wrist"},
        label="cameras",
    )
    front = required_table(data, "front")
    wrist = required_table(data, "wrist")
    _require_camera_keys(front, allow_depth=True, label="cameras.front")
    _require_camera_keys(wrist, allow_depth=False, label="cameras.wrist")
    config = SystemCamerasConfig(
        warmup_frames=integer(data, "warmup_frames"),
        max_age_s=floating(data, "max_age_s"),
        max_pair_skew_s=floating(data, "max_pair_skew_s"),
        front=_camera(front, require_square_crop=True, allow_depth=True),
        wrist=_camera(wrist, require_square_crop=False, allow_depth=False),
    )
    validate_system_cameras(config)
    return config


def validate_system_cameras(config: SystemCamerasConfig) -> None:
    if config.warmup_frames < 0:
        raise ValueError("cameras.warmup_frames must be non-negative")
    if config.max_age_s <= 0:
        raise ValueError("cameras.max_age_s must be positive")
    if not 0 < config.max_pair_skew_s <= config.max_age_s:
        raise ValueError(
            "cameras.max_pair_skew_s must be positive and no greater than max_age_s"
        )
    if config.front.crop is None:
        raise ValueError("system AgentView crop must be enabled")
    if (
        isinstance(config.front, SystemRealSenseCameraConfig)
        and config.front.depth
        and not config.front.align_depth_to_color
    ):
        raise ValueError(
            "AgentView depth must align to color when the required crop is enabled"
        )
    for name, camera in (("front", config.front), ("wrist", config.wrist)):
        if min(camera.width, camera.height, camera.fps) <= 0:
            raise ValueError(f"cameras.{name} dimensions/fps must be positive")
        if isinstance(camera, SystemRealSenseCameraConfig):
            if camera.depth:
                if camera.depth_width is None or camera.depth_height is None:
                    raise ValueError(f"cameras.{name} depth dimensions are required")
                if min(camera.depth_width, camera.depth_height) <= 0:
                    raise ValueError(
                        f"cameras.{name} depth dimensions must be positive"
                    )
            if not camera.auto_exposure:
                if camera.exposure is None or camera.gain is None:
                    raise ValueError(
                        f"cameras.{name} manual exposure and gain are required"
                    )
                if min(camera.exposure, camera.gain) < 0:
                    raise ValueError(
                        f"cameras.{name} exposure/gain must be non-negative"
                    )
            if not camera.auto_white_balance and (
                camera.white_balance is None or camera.white_balance < 0
            ):
                raise ValueError(
                    f"cameras.{name} manual white_balance must be non-negative"
                )
        elif isinstance(camera, SystemV4l2CameraConfig):
            if camera.backend_api not in {"v4l2", "any"}:
                raise ValueError(f"cameras.{name}.backend_api must be v4l2 or any")
            if camera.pixel_format and len(camera.pixel_format) != 4:
                raise ValueError(
                    f"cameras.{name}.pixel_format must be empty or four characters"
                )


def required_front_roi(config: SystemCamerasConfig) -> ImageRoi:
    """Return the already-validated square AgentView input region."""

    roi = config.front.crop
    if roi is None:  # Defensive for manually constructed configs.
        raise ValueError("system AgentView crop must be enabled")
    if roi.width != roi.height:
        raise ValueError(f"system AgentView crop must be square, got {roi.xywh}")
    return roi


def _require_camera_keys(
    data: dict[str, Any], *, allow_depth: bool, label: str
) -> None:
    common = {
        "backend",
        "width",
        "height",
        "fps",
        "crop_enabled",
    }
    if boolean(data, "crop_enabled"):
        common |= {"crop_x", "crop_y", "crop_width", "crop_height"}
    backend = string(data, "backend")
    if backend == "realsense":
        realsense = {
            "serial",
            "require_usb3",
            "auto_exposure",
            "auto_white_balance",
        }
        if not boolean(data, "auto_exposure"):
            realsense |= {"exposure", "gain"}
        if not boolean(data, "auto_white_balance"):
            realsense.add("white_balance")
        if allow_depth:
            realsense.add("depth")
            if boolean(data, "depth"):
                realsense |= {
                    "depth_width",
                    "depth_height",
                    "align_depth_to_color",
                }
        required = common | realsense
    elif backend == "v4l2":
        required = common | {"device", "backend_api", "pixel_format"}
    else:
        raise ValueError(f"{label}.backend must be realsense or v4l2")
    require_exact_keys(data, required=required, label=label)


def _camera(
    data: dict[str, Any], *, require_square_crop: bool, allow_depth: bool
) -> SystemCameraDeviceConfig:
    backend = string(data, "backend")
    width, height = integer(data, "width"), integer(data, "height")
    common = {
        "backend": backend,
        "width": width,
        "height": height,
        "fps": integer(data, "fps"),
        "crop": parse_optional_image_roi(
            data,
            image_width=width,
            image_height=height,
            label="system camera crop",
            require_square=require_square_crop,
        ),
    }
    if backend == "realsense":
        depth = boolean(data, "depth") if allow_depth else False
        auto_exposure = boolean(data, "auto_exposure")
        auto_white_balance = boolean(data, "auto_white_balance")
        return SystemRealSenseCameraConfig(
            **common,
            serial=string(data, "serial"),
            require_usb3=boolean(data, "require_usb3"),
            depth=depth,
            depth_width=integer(data, "depth_width") if depth else None,
            depth_height=integer(data, "depth_height") if depth else None,
            align_depth_to_color=(
                boolean(data, "align_depth_to_color") if depth else None
            ),
            auto_exposure=auto_exposure,
            exposure=None if auto_exposure else integer(data, "exposure"),
            gain=None if auto_exposure else integer(data, "gain"),
            auto_white_balance=auto_white_balance,
            white_balance=(
                None if auto_white_balance else integer(data, "white_balance")
            ),
        )
    return SystemV4l2CameraConfig(
        **common,
        device=string(data, "device"),
        backend_api=string(data, "backend_api"),
        pixel_format=text(data, "pixel_format"),
    )
