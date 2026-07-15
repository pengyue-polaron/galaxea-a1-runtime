"""Small pure helpers for A1 teleoperation data collection."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from galaxea_a1_runtime.collection.episode_output import validate_staged_episode
from galaxea_a1_runtime.schema import (
    ActionMode,
    DEFAULT_STATE_NAMES,
)

TELEOP_RAW_SCHEMA_VERSION = "galaxea_a1_teleop_raw_v3"
EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

EEF_STATE_NAMES = DEFAULT_STATE_NAMES[:7]
JOINT_STATE_NAMES = DEFAULT_STATE_NAMES[7:]
GRIPPER_STATE_NAME = DEFAULT_STATE_NAMES[-1:]


class StateMode(StrEnum):
    EEF = "eef"
    JOINT = "joint"
    EEF_JOINT = "eef_joint"


class EpisodeDecision(StrEnum):
    SAVE = "save"
    DISCARD = "discard"
    QUIT = "quit"


def reset_required_after_episode(
    decision: EpisodeDecision | str,
    *,
    after_save: bool,
    after_discard: bool,
) -> bool:
    """Return the tracked reset policy for a completed recording decision."""

    decision = EpisodeDecision(decision)
    if decision == EpisodeDecision.SAVE:
        return after_save
    if decision == EpisodeDecision.DISCARD:
        return after_discard
    return False


def validate_experiment_name(value: str) -> str:
    if value in {".", ".."} or EXPERIMENT_NAME.fullmatch(value) is None:
        raise ValueError(
            "experiment must be 1-128 characters using letters, digits, '.', '_', "
            "or '-', must start with a letter/digit, and cannot be '.' or '..'"
        )
    return value


@dataclass(frozen=True)
class CameraMetadata:
    name: str
    directory: str
    width: int
    height: int
    source: str | int | None = None
    modality: str = "rgb"
    dtype: str = "uint8"
    encoding: str = "bgr8_jpeg"
    source_width: int | None = None
    source_height: int | None = None
    crop_xywh: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class TeleopRawEpisodeMetadata:
    schema_version: str
    collection_mode: str
    task: str
    experiment: str
    episode_index: int
    frame_count: int
    fps_target: float
    state_mode: StateMode
    action_mode: ActionMode
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_topics: dict[str, str]
    action_topics: dict[str, str]
    control_path: tuple[str, ...]
    cameras: tuple[CameraMetadata, ...]
    config_path: str | None = None
    quality_checks: dict[str, float] = field(default_factory=dict)


def state_names_for_mode(mode: StateMode | str) -> tuple[str, ...]:
    mode = StateMode(mode)
    if mode == StateMode.EEF:
        return (*EEF_STATE_NAMES, *GRIPPER_STATE_NAME)
    if mode == StateMode.JOINT:
        return JOINT_STATE_NAMES
    if mode == StateMode.EEF_JOINT:
        return DEFAULT_STATE_NAMES
    raise ValueError(f"unsupported state mode: {mode}")


def state_columns(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"state.{name}" for name in names)


def action_columns(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"action.{name}" for name in names)


def teleop_frame_header(
    *,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    camera_dirs: tuple[str, ...] = ("cam0", "cam1"),
) -> tuple[str, ...]:
    camera_columns = tuple(f"{directory}_relpath" for directory in camera_dirs)
    return (
        "frame_index",
        "wall_time_ns",
        "ros_stamp_s",
        "cam0_seq",
        "cam0_monotonic_s",
        "cam1_seq",
        "cam1_monotonic_s",
        *camera_columns,
        *state_columns(state_names),
        *action_columns(action_names),
    )


def normalize_episode_decision(text: str | None) -> EpisodeDecision:
    value = (text or "").strip().lower()
    if value in {"d", "discard"}:
        return EpisodeDecision.DISCARD
    if value in {"q", "quit", "exit"}:
        return EpisodeDecision.QUIT
    return EpisodeDecision.SAVE


def next_episode_index(experiment_dir: Path) -> int:
    indices: list[int] = []
    for path in experiment_dir.glob("episode_*"):
        if not path.is_dir():
            continue
        parts = path.name.split("_")
        if len(parts) < 2:
            continue
        try:
            indices.append(int(parts[1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def validate_episode_layout(experiment_dir: Path) -> None:
    """Refuse collection beside crash leftovers or incomplete final episodes."""

    if not experiment_dir.exists():
        return
    staging = sorted(
        path.name
        for path in experiment_dir.iterdir()
        if path.name.startswith(".episode_") and ".staging-" in path.name
    )
    incomplete: list[str] = []
    for path in sorted(experiment_dir.glob("episode_*")):
        if not path.is_dir():
            continue
        try:
            metadata = json.loads((path / "metadata.json").read_text())
            frame_count = metadata.get("frame_count")
            if isinstance(frame_count, bool) or not isinstance(frame_count, int):
                raise RuntimeError(f"invalid metadata frame_count: {frame_count!r}")
            cameras = metadata.get("cameras")
            if not isinstance(cameras, list):
                raise RuntimeError("metadata cameras must be a list")
            depth_enabled = any(
                isinstance(camera, dict) and camera.get("directory") == "cam0_depth"
                for camera in cameras
            )
            validate_staged_episode(
                path,
                frame_count=frame_count,
                depth_enabled=depth_enabled,
            )
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError) as exc:
            incomplete.append(f"{path.name}({exc})")
    if staging or incomplete:
        details = [
            *(f"staging:{name}" for name in staging),
            *(f"incomplete:{name}" for name in incomplete),
        ]
        raise ValueError(
            "raw experiment contains uncommitted episode output; inspect and remove or "
            f"quarantine it before collecting: {details}"
        )


def metadata_to_json_dict(metadata: TeleopRawEpisodeMetadata) -> dict:
    data = asdict(metadata)
    data["state_mode"] = metadata.state_mode.value
    data["action_mode"] = metadata.action_mode.value
    data["state_names"] = list(metadata.state_names)
    data["action_names"] = list(metadata.action_names)
    data["control_path"] = list(metadata.control_path)
    return data


def validate_existing_camera_shape(
    experiment_dir: Path,
    *,
    camera_name: str,
    width: int,
    height: int,
) -> None:
    """Reject appending frames with a different shape to an existing raw experiment."""

    mismatches: list[str] = []
    for metadata_path in sorted(experiment_dir.glob("episode_*/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read existing episode metadata: {metadata_path}: {exc}"
            ) from exc
        cameras = payload.get("cameras")
        if not isinstance(cameras, list):
            raise ValueError(
                f"existing episode metadata has no camera list: {metadata_path}"
            )
        camera = next(
            (
                item
                for item in cameras
                if isinstance(item, dict) and item.get("name") == camera_name
            ),
            None,
        )
        if camera is None:
            raise ValueError(
                f"existing episode metadata has no {camera_name!r} camera: {metadata_path}"
            )
        existing = (int(camera.get("width", 0)), int(camera.get("height", 0)))
        if existing != (width, height):
            mismatches.append(
                f"{metadata_path.parent.name}={existing[0]}x{existing[1]}"
            )
    if mismatches:
        preview = ", ".join(mismatches[:3])
        suffix = " ..." if len(mismatches) > 3 else ""
        raise ValueError(
            f"cannot append {camera_name} {width}x{height} frames to an existing experiment "
            f"with a different camera shape ({preview}{suffix}); use a new experiment name "
            "or migrate all existing episodes to the tracked crop first"
        )


def validate_existing_schema(experiment_dir: Path, *, expected: str) -> None:
    """Reject appending the continuous contract to an older experiment directory."""

    mismatches: list[str] = []
    for metadata_path in sorted(experiment_dir.glob("episode_*/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read existing episode metadata: {metadata_path}: {exc}"
            ) from exc
        actual = str(payload.get("schema_version", "missing"))
        if actual != expected:
            mismatches.append(f"{metadata_path.parent.name}={actual}")
    if mismatches:
        preview = ", ".join(mismatches[:3])
        suffix = " ..." if len(mismatches) > 3 else ""
        raise ValueError(
            f"cannot append {expected} episodes to an experiment with another schema "
            f"({preview}{suffix}); use a new experiment name"
        )
