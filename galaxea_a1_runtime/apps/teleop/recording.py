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


@dataclass(frozen=True)
class RecordedEpisode:
    bag_root: Path
    elapsed_s: float
    decision: EpisodeDecision
    reset_required_override: bool | None = None


def record_episode(*, config, experiment, task, cameras, ros_state, on_ready):
    import rospy

    capture = BagCapture(config, experiment=experiment, task=task)
    decision = normalize_collection_recording_decision("q")
    outcome = "interrupted"
    try:
        capture.start()
        info(f"Raw episode: {capture.root}")
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
                detail=f"Recording original camera and robot streams · {elapsed:.1f}s",
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
