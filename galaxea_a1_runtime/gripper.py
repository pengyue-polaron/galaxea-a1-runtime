"""Shared continuous gripper normalization for collection and deployment."""

from __future__ import annotations

import math


def normalize_stroke(stroke_mm: float, *, stroke_min_mm: float, stroke_max_mm: float) -> float:
    """Map a physical stroke into the repository-wide normalized ``[0, 1]`` contract."""

    _validate_range(stroke_min_mm, stroke_max_mm)
    stroke = _finite(stroke_mm, label="gripper stroke")
    if stroke < stroke_min_mm - 1e-6 or stroke > stroke_max_mm + 1e-6:
        raise ValueError(
            f"gripper stroke {stroke:g}mm is outside configured range "
            f"[{stroke_min_mm:g}, {stroke_max_mm:g}]mm"
        )
    normalized = (stroke - stroke_min_mm) / (stroke_max_mm - stroke_min_mm)
    return min(1.0, max(0.0, normalized))


def denormalize_stroke(normalized: float, *, stroke_min_mm: float, stroke_max_mm: float) -> float:
    """Map a normalized continuous target into the configured physical stroke range."""

    _validate_range(stroke_min_mm, stroke_max_mm)
    value = min(1.0, max(0.0, _finite(normalized, label="normalized gripper target")))
    return stroke_min_mm + value * (stroke_max_mm - stroke_min_mm)


def _validate_range(stroke_min_mm: float, stroke_max_mm: float) -> None:
    lower = _finite(stroke_min_mm, label="gripper stroke minimum")
    upper = _finite(stroke_max_mm, label="gripper stroke maximum")
    if upper <= lower:
        raise ValueError("gripper stroke maximum must be greater than minimum")


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result
