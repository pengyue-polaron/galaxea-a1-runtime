"""Pure formatting and validation for the Teleop gripper readback tool."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperDebugReading:
    leader_position: float
    action_normalized: float
    target_mm: float
    feedback_mm: float
    state_normalized: float
    error_mm: float


def build_gripper_debug_reading(
    *,
    target_mm: float,
    feedback_mm: float,
    stroke_min_mm: float,
    stroke_max_mm: float,
    source_min: float,
    source_max: float,
    invert: bool,
) -> GripperDebugReading:
    values = (
        target_mm,
        feedback_mm,
        stroke_min_mm,
        stroke_max_mm,
        source_min,
        source_max,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gripper debug values must be finite")
    if stroke_max_mm <= stroke_min_mm:
        raise ValueError("gripper debug stroke range is invalid")
    if source_max <= source_min:
        raise ValueError("leader gripper debug source range is invalid")
    if not stroke_min_mm <= target_mm <= stroke_max_mm:
        raise ValueError(
            f"gripper target {target_mm:g}mm is outside "
            f"[{stroke_min_mm:g}, {stroke_max_mm:g}]mm"
        )
    action_normalized = (float(target_mm) - float(stroke_min_mm)) / (
        float(stroke_max_mm) - float(stroke_min_mm)
    )
    source_normalized = 1.0 - action_normalized if invert else action_normalized
    leader_position = float(source_min) + source_normalized * (
        float(source_max) - float(source_min)
    )
    state_normalized = (float(feedback_mm) - float(stroke_min_mm)) / (
        float(stroke_max_mm) - float(stroke_min_mm)
    )
    return GripperDebugReading(
        leader_position=leader_position,
        action_normalized=action_normalized,
        target_mm=float(target_mm),
        feedback_mm=float(feedback_mm),
        state_normalized=state_normalized,
        error_mm=float(feedback_mm) - float(target_mm),
    )


def format_gripper_debug_reading(
    reading: GripperDebugReading, *, relay_summary: str
) -> str:
    return (
        f"leader={reading.leader_position:7.3f}  "
        f"action={reading.action_normalized:6.3f}  "
        f"target={reading.target_mm:7.3f} mm  |  "
        f"A1={reading.feedback_mm:7.3f} mm  "
        f"state={reading.state_normalized:6.3f}  "
        f"delta={reading.error_mm:+7.3f} mm  |  relay={relay_summary}"
    )
