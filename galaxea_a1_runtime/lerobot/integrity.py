"""Galaxea constraints around the shared LeRobot v3 payload validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_ops.datasets.lerobot import (
    LEROBOT_GENERATED_FRAME_COLUMNS,
    validate_lerobot_v3_dataset,
)
from embodied_ops.datasets.lerobot import (
    read_lerobot_v3_tasks as _read_lerobot_v3_tasks,
)

from galaxea_a1_runtime.schema import ACTION_FEATURE_KEY, STATE_FEATURE_KEY


def validate_lerobot_v3_payloads(
    root: Path,
    *,
    info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    expected_tasks: tuple[str, ...],
) -> None:
    """Validate shared v3 mechanics plus the canonical Galaxea feature contract."""

    validate_lerobot_v3_dataset(
        root,
        info=info,
        expected_episodes=total_episodes,
        expected_frames=total_frames,
        expected_tasks=expected_tasks,
        required_frame_columns=(
            *LEROBOT_GENERATED_FRAME_COLUMNS,
            STATE_FEATURE_KEY,
            ACTION_FEATURE_KEY,
        ),
        required_stat_features=(STATE_FEATURE_KEY, ACTION_FEATURE_KEY),
    )


def read_lerobot_v3_tasks(root: Path, *, info: dict[str, Any]) -> tuple[str, ...]:
    """Return canonical task text ordered by LeRobot task index."""

    return _read_lerobot_v3_tasks(root, info=info)
