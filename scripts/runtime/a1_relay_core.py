#!/usr/bin/env python3
"""Compatibility shim for the ROS relay safety core.

The maintained implementation lives in `galaxea_a1_runtime.safety`. This file
keeps existing ROS scripts importable when they are executed directly from the
`scripts/runtime` directory inside Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.safety import (  # noqa: E402
    RelayInputs,
    relay_block_reason,
    validate_initial_alignment,
)

def validate_inputs(inputs: RelayInputs, *, arm_joints: int, max_age: float) -> str | None:
    return relay_block_reason(inputs, arm_joints=arm_joints, max_age=max_age)


def check_initial_alignment(
    current: Sequence[float],
    raw: Sequence[float],
    max_abs_error: float,
) -> None:
    validate_initial_alignment(current, raw, max_abs_error=max_abs_error)
