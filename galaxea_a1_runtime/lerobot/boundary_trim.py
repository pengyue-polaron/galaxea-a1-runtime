"""Conservative, auditable trimming of stationary episode boundaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from galaxea_a1_runtime.lerobot.boundary_trim_config import BoundaryTrimConfig

ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
GRIPPER_NAME = "gripper"
TRIM_FORMAT = "galaxea_a1_boundary_trim_v1"


@dataclass(frozen=True)
class EpisodeTrimDecision:
    """Half-open source frame bounds and the evidence used to select them."""

    start: int
    end: int
    original_frames: int
    start_anchor_stable: bool
    end_action_anchor_stable: bool
    end_state_anchor_stable: bool
    start_reason: str
    end_reason: str
    guard_reason: str | None = None

    @property
    def kept_frames(self) -> int:
        return self.end - self.start

    @property
    def prefix_trimmed_frames(self) -> int:
        return self.start

    @property
    def suffix_trimmed_frames(self) -> int:
        return self.original_frames - self.end

    @property
    def trimmed_fraction(self) -> float:
        if self.original_frames == 0:
            return 0.0
        return (self.original_frames - self.kept_frames) / self.original_frames

    def metadata_record(self, *, episode: str) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "episode": episode,
                "kept_frames": self.kept_frames,
                "prefix_trimmed_frames": self.prefix_trimmed_frames,
                "suffix_trimmed_frames": self.suffix_trimmed_frames,
                "trimmed_fraction": self.trimmed_fraction,
            }
        )
        return record


def decide_episode_bounds(
    *,
    actions: np.ndarray,
    states: np.ndarray,
    action_names: tuple[str, ...],
    state_names: tuple[str, ...],
    fps: int,
    config: BoundaryTrimConfig,
) -> EpisodeTrimDecision:
    """Return conservative ``[start, end)`` bounds without mutating input data."""

    action_values = _finite_matrix(actions, label="actions")
    state_values = _finite_matrix(states, label="states")
    if len(action_values) != len(state_values):
        raise ValueError("trim action/state frame counts differ")
    if fps <= 0:
        raise ValueError("trim FPS must be positive")

    action_motion = _motion_columns(action_values, names=action_names, label="action")
    state_motion = _motion_columns(state_values, names=state_names, label="state")
    frame_count = len(action_motion)
    if not config.enabled:
        return _whole_episode(
            frame_count,
            start_anchor_stable=False,
            end_action_anchor_stable=False,
            end_state_anchor_stable=False,
            start_reason="disabled",
            end_reason="disabled",
        )

    anchor_frames = math.ceil(config.anchor_window_s * fps)
    if frame_count < anchor_frames:
        return _whole_episode(
            frame_count,
            start_anchor_stable=False,
            end_action_anchor_stable=False,
            end_state_anchor_stable=False,
            start_reason="episode_shorter_than_anchor",
            end_reason="episode_shorter_than_anchor",
        )

    start_anchor = action_motion[:anchor_frames]
    end_action_anchor = action_motion[-anchor_frames:]
    end_state_anchor = state_motion[-anchor_frames:]
    start_anchor_stable = _anchor_is_stable(start_anchor, config=config)
    end_action_anchor_stable = _anchor_is_stable(end_action_anchor, config=config)
    end_state_anchor_stable = _anchor_is_stable(end_state_anchor, config=config)

    start = 0
    start_reason = "unstable_action_anchor"
    first_departure: int | None = None
    if start_anchor_stable:
        start_outside = _outside_reference(
            action_motion,
            reference=np.median(start_anchor, axis=0),
            config=config,
        )
        first_departure = _first_confirmed_true(
            start_outside, count=config.confirm_frames
        )
        if first_departure is None:
            start_reason = "no_confirmed_departure"
        else:
            pre_roll_frames = math.ceil(config.pre_roll_s * fps)
            start = max(0, first_departure - pre_roll_frames)
            start_reason = "confirmed_action_departure"

    end = frame_count
    end_reason = "unstable_action_anchor"
    if end_action_anchor_stable and end_state_anchor_stable:
        action_outside = _outside_reference(
            action_motion,
            reference=np.median(end_action_anchor, axis=0),
            config=config,
        )
        state_outside = _outside_reference(
            state_motion,
            reference=np.median(end_state_anchor, axis=0),
            config=config,
        )
        last_action_departure = _last_confirmed_true(
            action_outside, count=config.confirm_frames
        )
        last_state_departure = _last_confirmed_true(
            state_outside, count=config.confirm_frames
        )
        if last_action_departure is None:
            end_reason = "no_confirmed_action_departure"
        else:
            last_departure = max(
                value
                for value in (last_action_departure, last_state_departure)
                if value is not None
            )
            post_roll_frames = math.ceil(config.post_roll_s * fps)
            end = min(frame_count, last_departure + 1 + post_roll_frames)
            end_reason = "confirmed_action_and_feedback_settle"
    elif end_action_anchor_stable:
        end_reason = "unstable_feedback_anchor"

    if first_departure is None:
        return _whole_episode(
            frame_count,
            start_anchor_stable=start_anchor_stable,
            end_action_anchor_stable=end_action_anchor_stable,
            end_state_anchor_stable=end_state_anchor_stable,
            start_reason=start_reason,
            end_reason=end_reason,
            guard_reason="no_confirmed_task_motion",
        )

    removed_frames = start + (frame_count - end)
    kept_frames = end - start
    min_kept_frames = math.ceil(config.min_kept_duration_s * fps)
    guard_reason = None
    if kept_frames <= 0:
        guard_reason = "empty_or_overlapping_bounds"
    elif kept_frames < min_kept_frames:
        guard_reason = "minimum_kept_duration"
    elif removed_frames / frame_count > config.max_trim_fraction:
        guard_reason = "maximum_trim_fraction"
    if guard_reason is not None:
        return _whole_episode(
            frame_count,
            start_anchor_stable=start_anchor_stable,
            end_action_anchor_stable=end_action_anchor_stable,
            end_state_anchor_stable=end_state_anchor_stable,
            start_reason=start_reason,
            end_reason=end_reason,
            guard_reason=guard_reason,
        )

    return EpisodeTrimDecision(
        start=start,
        end=end,
        original_frames=frame_count,
        start_anchor_stable=start_anchor_stable,
        end_action_anchor_stable=end_action_anchor_stable,
        end_state_anchor_stable=end_state_anchor_stable,
        start_reason=start_reason,
        end_reason=end_reason,
    )


def trim_manifest(
    *,
    decisions: tuple[tuple[str, EpisodeTrimDecision], ...],
    fps: int,
    config: BoundaryTrimConfig,
) -> dict[str, Any]:
    original_frames = sum(decision.original_frames for _, decision in decisions)
    kept_frames = sum(decision.kept_frames for _, decision in decisions)
    prefix_frames = sum(decision.prefix_trimmed_frames for _, decision in decisions)
    suffix_frames = sum(decision.suffix_trimmed_frames for _, decision in decisions)
    return {
        "format": TRIM_FORMAT,
        "fps": fps,
        "policy": asdict(config),
        "summary": {
            "episodes": len(decisions),
            "original_frames": original_frames,
            "kept_frames": kept_frames,
            "trimmed_frames": original_frames - kept_frames,
            "prefix_trimmed_frames": prefix_frames,
            "suffix_trimmed_frames": suffix_frames,
            "trimmed_fraction": (
                (original_frames - kept_frames) / original_frames
                if original_frames
                else 0.0
            ),
            "guarded_episodes": sum(
                decision.guard_reason is not None for _, decision in decisions
            ),
        },
        "episodes": [
            decision.metadata_record(episode=episode) for episode, decision in decisions
        ],
    }


def _finite_matrix(values: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"trim {label} must be a two-dimensional matrix")
    if len(result) == 0:
        raise ValueError(f"trim {label} must contain at least one frame")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"trim {label} contain non-finite values")
    return result


def _motion_columns(
    values: np.ndarray, *, names: tuple[str, ...], label: str
) -> np.ndarray:
    if values.shape[1] != len(names):
        raise ValueError(f"trim {label} vector width does not match its names")
    if len(set(names)) != len(names):
        raise ValueError(f"trim {label} names contain duplicates")
    required = (*ARM_JOINT_NAMES, GRIPPER_NAME)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"trim {label} names are missing: {missing}")
    return values[:, [names.index(name) for name in required]]


def _anchor_is_stable(values: np.ndarray, *, config: BoundaryTrimConfig) -> bool:
    spread = np.ptp(values, axis=0)
    return bool(
        np.all(spread[:6] < config.joint_deadband_rad)
        and spread[6] < config.gripper_deadband
    )


def _outside_reference(
    values: np.ndarray,
    *,
    reference: np.ndarray,
    config: BoundaryTrimConfig,
) -> np.ndarray:
    deviation = np.abs(values - reference)
    return np.any(deviation[:, :6] >= config.joint_deadband_rad, axis=1) | (
        deviation[:, 6] >= config.gripper_deadband
    )


def _first_confirmed_true(values: np.ndarray, *, count: int) -> int | None:
    starts = _confirmed_run_starts(values, count=count)
    return int(starts[0]) if len(starts) else None


def _last_confirmed_true(values: np.ndarray, *, count: int) -> int | None:
    starts = _confirmed_run_starts(values, count=count)
    return int(starts[-1] + count - 1) if len(starts) else None


def _confirmed_run_starts(values: np.ndarray, *, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("trim confirm_frames must be positive")
    mask = np.asarray(values, dtype=bool)
    if len(mask) < count:
        return np.empty(0, dtype=np.int64)
    runs = np.convolve(mask.astype(np.int64), np.ones(count, dtype=np.int64), "valid")
    return np.flatnonzero(runs == count)


def _whole_episode(
    frame_count: int,
    *,
    start_anchor_stable: bool,
    end_action_anchor_stable: bool,
    end_state_anchor_stable: bool,
    start_reason: str,
    end_reason: str,
    guard_reason: str | None = None,
) -> EpisodeTrimDecision:
    return EpisodeTrimDecision(
        start=0,
        end=frame_count,
        original_frames=frame_count,
        start_anchor_stable=start_anchor_stable,
        end_action_anchor_stable=end_action_anchor_stable,
        end_state_anchor_stable=end_state_anchor_stable,
        start_reason=start_reason,
        end_reason=end_reason,
        guard_reason=guard_reason,
    )
