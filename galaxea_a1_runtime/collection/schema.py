"""Small pure helpers for A1 teleoperation data collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from galaxea_a1_runtime.schema import ActionMode, DEFAULT_STATE_NAMES, JOINT_ACTION_NAMES

TELEOP_RAW_SCHEMA_VERSION = "galaxea_a1_teleop_raw_v1"

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


def action_names_for_teleop(mode: ActionMode | str) -> tuple[str, ...]:
    mode = ActionMode(mode)
    if mode != ActionMode.JOINT_ABSOLUTE:
        raise ValueError(f"teleop collection currently records joint_absolute actions, got {mode}")
    return JOINT_ACTION_NAMES


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


def metadata_to_json_dict(metadata: TeleopRawEpisodeMetadata) -> dict:
    data = asdict(metadata)
    data["state_mode"] = metadata.state_mode.value
    data["action_mode"] = metadata.action_mode.value
    data["state_names"] = list(metadata.state_names)
    data["action_names"] = list(metadata.action_names)
    data["control_path"] = list(metadata.control_path)
    return data
