"""Small pure contracts for canonical A1 teleoperation collection."""

from __future__ import annotations

import re
from enum import StrEnum

EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def normalize_episode_decision(text: str | None) -> EpisodeDecision:
    value = (text or "").strip().lower()
    if value in {"d", "discard"}:
        return EpisodeDecision.DISCARD
    if value in {"q", "quit", "exit"}:
        return EpisodeDecision.QUIT
    return EpisodeDecision.SAVE
