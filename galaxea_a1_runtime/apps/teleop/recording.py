"""Guarded raw-bag capture; training conversion occurs after finalization."""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from embodied_ops.collection import require_fresh_sample, require_pair_skew
from embodied_ops.operator_panel import announce_progress
from galaxea_a1_runtime.apps.teleop.bag_capture import BagCapture
from galaxea_a1_runtime.apps.teleop.interaction import (
    normalize_collection_recording_decision,
)
from galaxea_a1_runtime.collection import EpisodeDecision
from galaxea_a1_runtime.console import info

if TYPE_CHECKING:
    from galaxea_a1_runtime.hardware.cameras import CameraReader


PREPARATION_STILLNESS_WINDOW_S = 0.3
PREPARATION_MAX_JOINT_DRIFT_RAD = 0.15
PREPARATION_MAX_JOINT_SPEED_RAD_S = 0.5


class PreparationMotionError(RuntimeError):
    """The A1 moved after recording was requested but before it began."""


@dataclass(frozen=True)
class RecordedEpisode:
    bag_root: Path
    elapsed_s: float
    decision: EpisodeDecision
    reset_required_override: bool | None = None


def record_episode(*, config, experiment, task, cameras, ros_state, on_ready):
    import rospy

    initial = _position_vector(ros_state)
    capture = BagCapture(config, experiment=experiment, task=task)
    decision = normalize_collection_recording_decision("q")
    outcome = "interrupted"
    try:
        capture.start()
        info(f"Raw episode: {capture.root}")
        try:
            _require_stationary_start(ros_state, initial)
        except PreparationMotionError:
            outcome = "discard"
            raise
        capture.begin()
        on_ready()
        start = time.monotonic()
        while not rospy.is_shutdown():
            capture.check()
            now = time.perf_counter()
            pair = cameras.camera_bridge.latest_pair()
            if pair is None:
                raise RuntimeError(
                    "synchronized camera pair unavailable during raw recording"
                )
            front, wrist = (
                require_fresh_sample(
                    s, label=label, now_s=now, max_age_s=config.system.cameras.max_age_s
                )
                for s, label in zip(pair, ("front", "wrist"), strict=True)
            )
            require_pair_skew(
                front,
                wrist,
                left_label="front",
                right_label="wrist",
                max_skew_s=config.system.cameras.max_pair_skew_s,
            )
            if ros_state.state_sample() is None or ros_state.action_values() is None:
                raise RuntimeError(
                    "robot state/action became stale during raw recording"
                )
            user_input = _poll_stdin_line()
            elapsed = time.monotonic() - start
            if user_input is not None or (
                config.collection.max_duration_s > 0
                and elapsed >= config.collection.max_duration_s
            ):
                decision = normalize_collection_recording_decision(user_input)
                outcome = decision.decision
                break
            announce_progress(
                "capture",
                "Raw episode capture",
                elapsed,
                config.collection.max_duration_s or None,
                phase="recording",
                detail="",
            )
            time.sleep(1.0 / config.collection.fps)
    finally:
        capture.stop(outcome)
    return RecordedEpisode(
        capture.root,
        (capture.manifest["end_ns"] - capture.manifest["start_ns"]) / 1e9,
        decision.decision,
        decision.reset_required_override,
    )


def wait_for_new_camera_samples(
    readers: tuple[CameraReader, ...],
    *,
    min_seq: dict[str, int],
    timeout_s: float,
) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        _raise_camera_reader_errors(readers)
        ready = True
        for reader in readers:
            latest = reader.latest()
            if latest is None or latest.seq <= min_seq.get(reader.name, -1):
                ready = False
                break
        if ready:
            return
        time.sleep(0.005)
    details = ", ".join(
        f"{reader.name}:seq={reader.latest_seq()}" for reader in readers
    )
    raise RuntimeError(
        f"camera readers did not produce fresh frames within {timeout_s:.1f}s ({details})"
    )


def _raise_camera_reader_errors(readers: tuple[CameraReader, ...]) -> None:
    for reader in readers:
        exc = reader.exception()
        if exc is not None:
            raise RuntimeError(f"{reader.name} camera reader failed") from exc


def _poll_stdin_line() -> str | None:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    line = sys.stdin.readline()
    if line == "":
        return "q"
    return line.strip().lower()


def _position_vector(ros_state) -> tuple[float, ...] | None:
    sample = ros_state.state_sample()
    if sample is None:
        return None
    return tuple(sample.values[7:14])


def _require_stationary_start(ros_state, initial: tuple[float, ...] | None) -> None:
    """Reject a start when the operator already moved the A1 during preparation."""

    if initial is None:
        raise RuntimeError("joint feedback unavailable before recording start")
    before = _position_vector(ros_state)
    time.sleep(PREPARATION_STILLNESS_WINDOW_S)
    current = _position_vector(ros_state)
    if before is None or current is None:
        raise RuntimeError("joint feedback became unavailable before recording start")
    drift = max(
        abs(value - origin) for value, origin in zip(current, initial, strict=True)
    )
    speed = (
        max(
            abs(value - previous)
            for value, previous in zip(current, before, strict=True)
        )
        / PREPARATION_STILLNESS_WINDOW_S
    )
    if (
        drift > PREPARATION_MAX_JOINT_DRIFT_RAD
        or speed > PREPARATION_MAX_JOINT_SPEED_RAD_S
    ):
        raise PreparationMotionError(
            "The A1 moved while recording was preparing, so the episode was not "
            "started. Keep the arm still until the console shows Recording."
        )
